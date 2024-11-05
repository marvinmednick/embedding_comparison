from django.contrib import admin
import logging
import tiktoken

from .models import Question, Choice
from .models import Document, Section
from .models import Patent, PatentClaim, ModifiedClaim, ClaimElement, ClaimRelatedSection, ModifiedClaimChunk
from .models import Embedding, ModificationType, EmbeddingType, Embedding768, Embedding32, Embedding1536
from .models import SectionChunkInfo, ModifiedSection, ClaimChunkInfo, ModifiedSectionChunk

logger = logging.getLogger(__name__)
logger.debug("Logger Test")

admin.site.register(Question)
admin.site.register(Choice)
admin.site.register(Embedding)
admin.site.register(ModificationType)
admin.site.register(Patent)


class DocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['id', 'name']


class ModifiedSectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'item', 'item_info', 'modification_type', 'item__document']
    readonly_fields = ['item_info']
    search_fields = ['id', 'modification_type__name']
    list_filter = ['modification_type', 'item__document']

    def item_info(self, obj):
        return obj.item.section_id

#     def section_id(self, obj):
#         return obj.section.section_id
# 
#     def mod_type(self, obj):
#         return obj.modification_type.name
# 
#     mod_type.short_description = 'Modification Type'


class ModifiedClaimAdmin(admin.ModelAdmin):
    list_display = ['id', 'item', 'modification_type']


class ModifiedSectionChunkAdmin(admin.ModelAdmin):
   
    list_display = ['id',  'mod_item_id', 'embed_type_name', 'chunk_number']
    readonly_fields = ['mod_item_id', 'embed_type_name']
    # search_fields = ['id', 'chunk_number']

    def mod_item_id(self, obj):
        return obj.modified_item_id

    def embed_type_name(self, obj):
        return obj.chunk_info.embed_type.name


class ModifiedClaimChunkAdmin(admin.ModelAdmin):
    list_display = ['id', 'modified_item', 'chunk_number']
    search_fields = ['id', 'modified_item__id']


class ClaimElementAdmin(admin.ModelAdmin):
    list_display = ['id', 'claim_id', 'element_number', 'text']
    search_fields = ['id', 'claim_id']


class SectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'section_id', 'document_name']
    readonly_fields = ['document_id', 'document_name']
    search_fields = ['id', 'section_id']
    list_filter = ['document__id']

    def document_name(self, obj):
        return obj.document.name

    def document_id(self, obj):
        return obj.document.id


class PatentClaimAdmin(admin.ModelAdmin):
    list_display = ['id', 'claim_id', 'related_claim']
    list_filter = ['related_claim']
    search_fields = ['id', 'claim_id']
    readonly_fields = ['id']


class ClaimChunkInfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'embed_type__short_name', 'source__item__claim_id', 'total_chunks']
    list_filter = ['embed_type__short_name', 'total_chunks']
    search_fields = ['id', 'source__item__claim_id']


class SectionChunkInfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'embed_type__short_name', 'source__item__section_id', 'total_chunks']
    list_filter = ['embed_type__short_name', 'total_chunks']
    search_fields = ['id', 'source__item__section_id']


admin.site.register(PatentClaim, PatentClaimAdmin)


class EmbeddingTypeAdmin(admin.ModelAdmin):
    list_display = ['id', 'short_name', 'name']


class EmbeddingAdmin(admin.ModelAdmin):
    list_display = ["id", "chunk_info_id", "chunk_info", "embed_source", "embed_type_shortname", 'mod_type_name', 'original_source', 'orig_source_id', "short_orig_text"]
    readonly_fields = ["id", "chunk_info_id", "source_id", 'orig_source_id', 'chunk_text', 'original_text', 'modified_text', 'original_source']
    list_filter = ['embed_type_shortname', 'total_chunks', 'embed_source', 'mod_type_name']
    search_fields = ['id', 'chunk_info_id', 'source_id', 'orig_source_id']

    def chunk_text(self, obj):
        if obj.embed_source == 'document':
            if obj.total_chunks == 1 or obj.chunk_number == 9999:
                chunk_text = ModifiedSection.objects.get(pk=obj.source_id).modified_text
            else:
                chunk_text = ModifiedSectionChunk.objects.get(chunk_info=obj.chunk_info_id, chunk_number=obj.chunkc_number).chunk_text

        elif obj.embed_source == 'claim':
            if obj.total_chunks == 1 or obj.chunk_number == 9999:
                chunk_text = ModifiedClaim.objects.get(pk=obj.source_id).modified_text
            else:
                chunk_text = ModifiedClaimChunk.objects.get(chunk_info=obj.chunk_info_id, chunk_number=obj.chunk_number).chunk_text

        else:
            chunk_text = "<Not Available>"
        return chunk_text

    def modified_text(self, obj):
        if obj.embed_source == 'document':
            mod_text = ModifiedSection.objects.get(pk=obj.source_id).modified_text
        elif obj.embed_source == 'claim':
            mod_text = ModifiedClaim.objects.get(pk=obj.orig_source_id).modified_text
        else:
            mod_text = "<Not Available>"

        return mod_text

    def original_text(self, obj):
        if obj.embed_source == 'document':
            orig_text = Section.objects.get(pk=obj.orig_source_id).text
        elif obj.embed_source == 'claim':
            orig_text = PatentClaim.objects.get(pk=obj.orig_source_id).text
        else:
            orig_text = "<Not Available>"

        return orig_text

    def short_orig_text(self, obj):
        return self.original_text(obj)[0:80]

    def original_source(self, obj):
        if obj.embed_source == 'document':
            orig_source = Section.objects.get(pk=obj.orig_source_id).section_id
        elif obj.embed_source == 'claim':
            orig_source = PatentClaim.objects.get(pk=obj.orig_source_id).claim_id
        else:
            orig_source = "<NA>"

        return orig_source

    def chunk_info(self, obj):
        return f"{obj.chunk_number}/{obj.total_chunks}"
    original_text.short_description = 'Source'


class ClaimRelatedSectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'claim__claim_id', 'related_sections']
    search_fields = ['id', 'claim__claim_id']


admin.site.register(Embedding32, EmbeddingAdmin)
admin.site.register(Embedding768, EmbeddingAdmin)
admin.site.register(Embedding1536, EmbeddingAdmin)
admin.site.register(EmbeddingType, EmbeddingTypeAdmin)
admin.site.register(ClaimChunkInfo, ClaimChunkInfoAdmin)
admin.site.register(SectionChunkInfo, SectionChunkInfoAdmin)
admin.site.register(Section, SectionAdmin)
admin.site.register(ClaimElement, ClaimElementAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(ModifiedSection, ModifiedSectionAdmin)
admin.site.register(ClaimRelatedSection, ClaimRelatedSectionAdmin)
admin.site.register(ModifiedSectionChunk, ModifiedSectionChunkAdmin)
admin.site.register(ModifiedClaim, ModifiedClaimAdmin)
admin.site.register(ModifiedClaimChunk, ModifiedClaimChunkAdmin)
