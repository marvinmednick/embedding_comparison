import os
import sys
import argparse
import django
import numpy as np
import logging
from tqdm import tqdm
from django.db.models import  Q

from log_setup import setup_logging, get_logger, switch_to_handler, update_log_levels
from utils import create_size_buckets, increment_bucket, comma_separated_list


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from sbert_embedding import SbertPatentEmbedding
from openai_embedding import OpenAIEmbedding
from topic_model import TopicModelEmbedding, display_models, get_full_model_name
from polls.models import ModifiedClaim, ClaimChunkInfo, ModifiedClaimChunk
from polls.models import ModifiedSection, SectionChunkInfo, ModifiedSectionChunk
from polls.models import ModificationType, EmbeddingType

# setup logging for this file
setup_logging()

logger = get_logger(__name__)


def chunk_doc(embedder, doctype, modtype, item_range=None, maxrec=None, item_list=None, token_check=False):

    doctype_config = {
        'doc': {
                'textObject':       ModifiedSection,
                'chunkObject':      ModifiedSectionChunk,
                'chunkInfoObject':  SectionChunkInfo,
                'embed_source':     'document',
                'status_desc':      'Section',
                'status_unit':      'section'
        },
        'claim': {
                'textObject':       ModifiedClaim,
                'chunkObject':      ModifiedClaimChunk,
                'chunkInfoObject':  ClaimChunkInfo,
                'embed_source':     'claim',
                'status_desc':      'Claim',
                'status_unit':      'claim'
        }
    }

    config = doctype_config[doctype]

    lookup_params, defaults, model = embedder.get_embed_params()

    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)

    mod_items_Q = Q(modification_type=modification_type)

    if item_list is not None:
        mod_items_Q &= Q(pk__in=item_list)

    items = config['textObject'].objects.filter(mod_items_Q)

    if item_range is not None:

        start, stop = item_range
        if maxrec is not None and (stop - start) > maxrec:
            stop = start + maxrec
            print(f"Shortening range to {maxrec} records:  {start}-{stop}")
        items = items[start:stop]

    elif maxrec is not None:
        items = items[0:maxrec]

    num_items = items.count()
    logger.debug("Processing %d %ss", num_items, config['status_desc'])
    logger.trace("Tracing")
    with tqdm(total=num_items, desc=f"Processing {config['status_desc']}", unit=config['status_unit']) as pbar:
        for index, item in enumerate(items):

            # Delete Embeddings that are for modified_text entry and embedding type (?)
            existing_embeddings = model.objects.filter(source_id=item.id, embed_source=config['embed_source'])
            existing_embedding_count = existing_embeddings.count()
            logger.debug("Deleting %d existing embedding records", existing_embedding_count)
            existing_embeddings.delete()

            # Delete ModifiedSectionChunk records for this modified section
            existing_chunks = config['chunkObject'].objects.filter(modified_item__id=item.id)
            existing_chunk_count = existing_chunks.count()
            logger.debug("Deleting %d existing chunk records", existing_chunk_count)
            existing_chunks.delete()

            chunked_text, total_chunks = embedder.chunk(item.modified_text)

            lookup_params = {
                'source': item,
                'embed_type': embedding_type
            }
            defaults = {
                'total_chunks': total_chunks
            }
            chunkInfoRef, _created = config['chunkInfoObject'].objects.update_or_create(**lookup_params, defaults=defaults)

            chunk_embed_list = []
            embed_record = {
                'embed_source': config['embed_source'],
                'chunk_info_id': chunkInfoRef.id,
                'source_id': chunkInfoRef.source.id,
                'orig_source_id': chunkInfoRef.source.item.id,
                'embed_type': embedding_type,
                'embed_type_name': embedding_type.name,
                'mod_type_name': chunkInfoRef.source.modification_type.name,
                'embed_type_shortname': embedding_type.short_name,
                'embedding_vector': None,
                'chunk_number': None,
                'total_chunks': total_chunks,
            }

            if total_chunks > 1:
                logger.debug("%s %s split into %d chunks", config['status_desc'], item.id, total_chunks)
            for chunk_num, chunk in enumerate(chunked_text, 1):

                # if there is more than one chunk, then create separate chunk records
                # otherwise the entire text is the chunk and the source text is the mod_text field in the modfiedsection record
                if total_chunks > 1:
                    chunk_info = {
                        'chunk_info': chunkInfoRef,
                        'modified_item': item,
                        'chunk_number': chunk_num,
                        'chunk_text': chunk
                    }
                    config['chunkObject'].objects.create(**chunk_info)

                chunk_embedding = embedder.generate_embedding(chunk)
                if total_chunks > 1:
                    logger.debug("Embed chunk #:%d  %s", chunk_num, str(chunk_embedding[0:6]))
                else:
                    logger.debug("%s %s single chunk - Embed:  %s", config['status_desc'], item.id,  str(chunk_embedding[0:6]))

                embed_record['chunk_number'] = chunk_num
                embed_record['embedding_vector'] = chunk_embedding

                chunk_embed_list.append(chunk_embedding)

                try:
                    model.objects.create(**embed_record)
                except Exception as e:
                    print(f"Exception {e} while creating embed record {embed_record}")
                    raise

            # if there is more than one chunk for the text,
            # also create a mean (otherwise only one chunk, there is no need for the mean)
            if total_chunks > 1:
                mean_embedding = np.mean([chunk_embedding for chunk_embedding in chunk_embed_list], axis=0)

                embed_record['chunk_number'] = 9999
                embed_record['embedding_vector'] = mean_embedding

                model.objects.create(**embed_record)
            pbar.update(1)


def embed_doc(embedder, modtype, maxrec=None, token_check=False):

    # ensure_specific_nltk_resources()

    lookup_params, defaults, model = embedder.get_embed_params()

    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        sections = ModifiedSection.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        sections = ModifiedSection.objects.filter(modification_type=modification_type)

    token_lengths = []

    # Create buckes for counting the size distribution of the token lengths
    specified_sizes = [16, 256, 512, 1024, 2048, 4096, 8192, 16384]
    size_buckets = create_size_buckets(specified_sizes)

    num_sections = sections.count()
    with tqdm(total=num_sections, desc="Processing Sections", unit="section") as pbar:
        for index, sec in enumerate(sections):
            tokens = embedder.tokenize(sec.modified_text)
            token_length = len(tokens['input_ids'])

            tl_rec = {
                'id': sec.id,
                'section_id': sec.item.section_id,
                'token_length': token_length,
                'text_length': sec.modified_text
            }
            token_lengths.append(tl_rec)

            increment_bucket(size_buckets, token_length)

            if not token_check:
                sec_embed_ref, _created = SectionChunkInfo.objects.update_or_create(source=sec, embed_type=embedding_type)

                text_embedding = embedder.generate_embedding(sec.modified_text)
                params = {
                    'embed_source': 'document',
                    'embed_id': sec_embed_ref.id,
                }
                defaults = {
                    'source_id': sec_embed_ref.source.id,
                    'orig_source_id': sec_embed_ref.source.item.id,
                    'embed_type': embedding_type,
                    'embed_type_name': embedding_type.name,
                    'mod_type_name': sec_embed_ref.source.modification_type.name,
                    'embed_type_shortname': embedding_type.short_name,
                    'embedding_vector': text_embedding,
                }
                model.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)

    for ln in token_lengths:
        print(f"Section {ln['section_id']} ({ln['id']})   len: {ln['token_length']}")

    print("Token sizes by range:")
    for key in size_buckets['size_list']:
        print(f"{key}: {size_buckets[key]}")

    print(f"max: {size_buckets['max']}")


def embed_patent_claims(embedder, modtype, maxrec=None, token_check=False):

    lookup_params, defaults, model = embedder.get_embed_params()
    embedding_type, _created = EmbeddingType.objects.get_or_create(**lookup_params, defaults=defaults)

    modification_type = ModificationType.objects.get(name=modtype)
    if maxrec is not None:
        claims = ModifiedClaim.objects.filter(modification_type=modification_type)[0:maxrec]
    else:
        claims = ModifiedClaim.objects.filter(modification_type=modification_type)

    num_claims = claims.count()
    with tqdm(total=num_claims, desc="Processing Claims", unit="claim") as pbar:
        for index, claim in enumerate(claims):
            claim_embed_ref, _created = ClaimChunkInfo.objects.update_or_create(source=claim, embed_type=embedding_type)
            text_embedding = embedder.generate_embedding(claim.modified_text)
            params = {
                'embed_source': 'claim',
                'embed_id': claim_embed_ref.id,
            }
            defaults = {
                'source_id': claim_embed_ref.source.id,
                'orig_source_id': claim_embed_ref.source.item.id,
                'embed_type': embedding_type,
                'embed_type_name': embedding_type.name,
                'mod_type_name': claim_embed_ref.source.modification_type.name,
                'embed_type_shortname': 'psbert',
                'embedding_vector': text_embedding,
            }
            model.objects.update_or_create(defaults=defaults, **params)
            pbar.update(1)
            logger.debug("Embed claim #:%d  %s", defaults['source_id'], str(text_embedding[0:6]))


def main():

    mod_types = ModificationType.objects.values_list('name', flat=True)
    embed_types = ['sbert', 'openai', 'topic']

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a collection and filename.")
    parser.add_argument('content', choices=['claims', 'doc', 'olddoc'], help='The name of the file to load.')
    parser.add_argument('embedtype', choices=embed_types, help='The type of embedding to use.')
    parser.add_argument('modtype', choices=mod_types, help='The text modificaiton to apply.')
    parser.add_argument('--startrec', type=int, default=0, help='starting record to process on loading, default is None (start at 0)')
    parser.add_argument('-r', '--range', nargs=2, type=int, metavar=('START', 'STOP'), help='Start and stop indices (inclusive)')
    parser.add_argument('--maxrec', type=int, default=None, help='maximum records to process on loading, default is None (use all)')
    parser.add_argument('--tokencheck', '-tc', action='store_true', help='Perform a token count check only')
    parser.add_argument('--model', type=str, help='Top Level Topic Model to use')
    parser.add_argument('--list_topic_models', action='store_true', help='Top Level Topic Model to use')
    parser.add_argument('--items', type=comma_separated_list, help="List of comma-separated modified (claim/section) numbers")
    parser.add_argument('--log-level', default=os.environ.get('LOG_LEVEL', 'INFO'),
                        choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--log-output', choices=['file', 'console'], help='Where to send log output to (default is file)')
    parser.add_argument('--list-loggers', action='store_true', help="Display a list of all loggers")

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # Display help if no command is provided or if there's an error
        parser.print_help()
        sys.exit()

    if args.list_topic_models:
        display_models()
        return

    if args.list_loggers:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for lg in loggers:
            print(lg)

    item_list = None
    if args.items:
        item_list = args.items

    if args.log_output:
        switch_to_handler(args.log_output)

    log_level = args.log_level.upper()
    if logger:
        update_log_levels(log_level)

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # No warning on sample size

    if args.embedtype == 'sbert':
        embedder = SbertPatentEmbedding()
    elif args.embedtype == 'openai':
        embedder = OpenAIEmbedding()
    elif args.embedtype == 'topic':
        model_name = get_full_model_name(args.model)
        if not model_name:
            print("topic embedding type requires a model to be set with --model")
            parser.print_help()
            sys.exit()

        # topic model requires a document to setup the model (and in particular
        # select a sub model based the entire document -- will use the entire document
        # for this selection even if max rec is set
        modification_type = ModificationType.objects.get(name=args.modtype)
        print("Gathering Sections")
        sections = [sec.modified_text for sec in ModifiedSection.objects.filter(modification_type=modification_type)]
        print("Creating Embbeder Model")
        embedder = TopicModelEmbedding(args.model, sections)

    else:
        print(f"Unknown embeddding type: {args.embed_type}")

    if args.content == 'claims':
        # embed_patent_claims(embedder, args.modtype, args.maxrec, args.tokencheck)
        chunk_doc(embedder, 'claim', args.modtype, args.range, args.maxrec, item_list, args.tokencheck)
    elif args.content == 'doc':
        chunk_doc(embedder, 'doc', args.modtype, args.range, args.maxrec, item_list, args.tokencheck)
    elif args.content == 'olddoc':
        embed_doc(embedder, args.modtype, args.maxrec, args.tokencheck)


if __name__ == "__main__":
    main()
