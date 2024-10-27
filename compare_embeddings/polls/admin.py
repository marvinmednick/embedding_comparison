from django.contrib import admin
import logging
import tiktoken

from .models import Question, Choice
from .models import Document, DocSection
from .models import Patent, PatentClaim, ClaimForEmbedding, ClaimElement, ClaimRelatedSection
from .models import Embedding, ModificationType, EmbeddingType, Embedding768, Embedding32, Embedding1536
from .models import SectionEmbedding, SectionForEmbedding, ClaimEmbedding

logger = logging.getLogger(__name__)
logger.debug("Logger Test")

admin.site.register(Question)
admin.site.register(Choice)
admin.site.register(Embedding)
admin.site.register(ModificationType)
admin.site.register(Patent)
admin.site.register(ClaimForEmbedding)


class DocumentAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['id', 'name']


class SectionForEmbeddingAdmin(admin.ModelAdmin):
    list_display = ['id', 'section', 'modification_type', 'section__document']
    # readonly_fields = ['section_id', 'mod_type']
    search_fields = ['id', 'section__section_id', 'modification_type__name']
    list_filter = ['modification_type', 'section__document']

#     def section_id(self, obj):
#         return obj.section.section_id
# 
#     def mod_type(self, obj):
#         return obj.modification_type.name
# 
#     mod_type.short_description = 'Modification Type'


class ClaimElementAdmin(admin.ModelAdmin):
    list_display = ['id', 'claim_id', 'element_number', 'text']
    search_fields = ['id', 'claim_id']


class DocSectionAdmin(admin.ModelAdmin):
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


class ClaimEmbeddingAdmin(admin.ModelAdmin):
    list_display = ['id', 'embed_type__short_name', 'source__claim__claim_id']
    list_filter = ['embed_type__short_name']
    search_fields = ['id', 'source__claim__claim_id']


class SectionEmbeddingAdmin(admin.ModelAdmin):
    list_display = ['id', 'embed_type__short_name', 'source__section__section_id']
    list_filter = ['embed_type__short_name']
    search_fields = ['id', 'source__section__section_id']


admin.site.register(PatentClaim, PatentClaimAdmin)


class EmbeddingTypeAdmin(admin.ModelAdmin):
    list_display = ['id', 'short_name', 'name']


class EmbeddingAdmin(admin.ModelAdmin):
    list_display = ["id", "embed_id", "embed_source", "embed_type_shortname", 'mod_type_name', 'original_source', 'orig_source_id', "short_orig_text"]
    readonly_fields = ["id", "embed_id", "source_id", 'orig_source_id', 'original_text', 'modified_text', 'original_source']
    list_filter = ['embed_type_shortname', 'embed_source', 'mod_type_name']
    search_fields = ['id', 'embed_id', 'source_id', 'orig_source_id']

    def modified_text(self, obj):
        if obj.embed_source == 'document':
            mod_text = SectionForEmbedding.objects.get(pk=obj.source_id).modified_text
        elif obj.embed_source == 'claim':
            mod_text = ClaimForEmbedding.objects.get(pk=obj.orig_source_id).modified_text
        else:
            mod_text = "<Not Available>"

        return mod_text

    def original_text(self, obj):
        if obj.embed_source == 'document':
            orig_text = DocSection.objects.get(pk=obj.orig_source_id).text
        elif obj.embed_source == 'claim':
            orig_text = PatentClaim.objects.get(pk=obj.orig_source_id).text
        else:
            orig_text = "<Not Available>"

        return orig_text

    def short_orig_text(self, obj):
        return self.original_text(obj)[0:80]

    def original_source(self, obj):
        if obj.embed_source == 'document':
            orig_source = DocSection.objects.get(pk=obj.orig_source_id).section_id
        elif obj.embed_source == 'claim':
            orig_source = PatentClaim.objects.get(pk=obj.orig_source_id).claim_id
        else:
            orig_source = "<NA>"

        return orig_source

    original_text.short_description = 'Source'


class ClaimRelatedSectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'claim__claim_id', 'related_sections']
    search_fields = ['id', 'claim__claim_id']


admin.site.register(Embedding32, EmbeddingAdmin)
admin.site.register(Embedding768, EmbeddingAdmin)
admin.site.register(Embedding1536, EmbeddingAdmin)
admin.site.register(EmbeddingType, EmbeddingTypeAdmin)
admin.site.register(ClaimEmbedding, ClaimEmbeddingAdmin)
admin.site.register(SectionEmbedding, SectionEmbeddingAdmin)
admin.site.register(DocSection, DocSectionAdmin)
admin.site.register(ClaimElement, ClaimElementAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(SectionForEmbedding, SectionForEmbeddingAdmin)
admin.site.register(ClaimRelatedSection, ClaimRelatedSectionAdmin)
