import os
import django
from django.db.models import Count
from tqdm import tqdm
from utils import get_embed_model

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()

from polls.models import SectionChunkInfo, ClaimChunkInfo, Tag  # noqa E402


def update_average_tags():

    result = SectionChunkInfo.objects.values('embed_type', 'embed_type__short_name').annotate(count=Count('embed_type')).order_by('embed_type')
    print(result)

    r = result.first()
    print(r['embed_type'], r['embed_type__short_name'])

    print(result)
    for item in result:
        print(f"Embed Type {item['embed_type__short_name']} - Count: {item['count']}")
        queryset_embed_type = SectionChunkInfo.objects.filter(embed_type__pk=item['embed_type'])
        print("\ttotal: ", queryset_embed_type.count())
        print("\tSingle chunk: ", queryset_embed_type.filter(total_chunks=1).count())
        print("\tMulti chunk: ", queryset_embed_type.exclude(total_chunks=1).count())

    print("\nOverall")
    queryset_single = SectionChunkInfo.objects.filter(total_chunks=1)
    queryset_multi = SectionChunkInfo.objects.exclude(total_chunks=1)
    print("\tSingle chunk: ", queryset_multi.count())
    print("\tMulti chunk: ", queryset_single.count())

    average_tag, _created = Tag.objects.get_or_create(name='average')
    print(f"Average tag Created: {_created}")

    queryset_all_sections = SectionChunkInfo.objects.all()
    queryset_all_claims = ClaimChunkInfo.objects.all()

    print("Staring Sections")
    with tqdm(total=queryset_all_sections.count(), desc="Processing Sections", unit="section") as pbar:
        for r in queryset_all_sections:
            embedding_model = get_embed_model(r.embed_type.size)

            embedding_queryset = embedding_model.objects.filter(chunk_info_id=r.id, embed_source='document')

            for f in embedding_queryset:
                average = (f.total_chunks == 1 and f.chunk_number == 1) or (f.total_chunks > 1 and f.chunk_number == 9999)
                if average:
                    f.tags.add(average_tag)

            pbar.update(1)

    print("Staring Claims")
    subset = queryset_all_claims[0:10]

    with tqdm(total=subset.count(), desc="Processing Claims", unit="claim") as pbar:
        for r in subset:
            #            print(f"\nChunk Info :  id:  {r.id} type: {r.embed_type} total chunks: {r.total_chunks} size: {r.embed_type.size}")

            embedding_model = get_embed_model(r.embed_type.size)

            embedding_queryset = embedding_model.objects.filter(chunk_info_id=r.id, embed_source='claim')

            # print("Embeddings")
            # print(f"Len of embedding_queryset: {len(embedding_queryset)}  {len(embedding_queryset)}")
            for f in embedding_queryset:
                average = (f.total_chunks == 1 and f.chunk_number == 1) or (f.total_chunks > 1 and f.chunk_number == 9999)
                # print(f"id: {f.id}, Chunk Info id {f.chunk_info_id}  source: {f.source_id}, orig Source: {f.orig_source_id}  embed source: {f.embed_source} {f.chunk_number}/{f.total_chunks} {average}")
                if average:
                    f.tags.add(average_tag)

            pbar.update(1)


def main():
    update_average_tags()


if __name__ == "__main__":
    main()
