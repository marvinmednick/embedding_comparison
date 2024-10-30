import datetime
from django.db import models
from django.utils import timezone
from pgvector.django import VectorField
from django.contrib.postgres.fields import ArrayField


class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published")

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class Embedding(models.Model):
    text = models.TextField(help_text="Text to be embedded")
    embedding = VectorField(
        dimensions=4,
        help_text="Vector embedding",
        null=True,
        blank=True
    )


class ModificationType(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return f"{self.name} ({self.pk})"


class EmbeddingType(models.Model):
    name = models.CharField(max_length=200)
    short_name = models.CharField(max_length=20, null=True)
    description = models.TextField()
    size = models.IntegerField()
    windows_size = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.short_name}: {self.name} ({self.pk})"


class Document(models.Model):
    name = models.CharField(max_length=200, help_text="Name of Document")
    filename = models.TextField(help_text="Name of Document")

    def __str__(self):
        return f"{self.name} ({self.pk})"


class Section(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    section_id = models.CharField(max_length=100)
    section_title = models.TextField(default="")
    section_title_text = models.TextField(default="")
    text = models.TextField()

    def __str__(self):
        return f"{self.section_id} ({self.pk})"


class ModifiedSection(models.Model):
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    modification_type = models.ForeignKey(ModificationType, on_delete=models.CASCADE)
    modified_text = models.TextField()
    openai_tokens = models.IntegerField(null=True)

    def __str__(self):
        return f"{self.section.section_id} - {self.modification_type.name}  ({self.pk})"


class SectionEmbedding(models.Model):
    embed_type = models.ForeignKey(EmbeddingType, on_delete=models.CASCADE)
    source = models.ForeignKey(ModifiedSection, on_delete=models.CASCADE)
    total_chunks = models.IntegerField(default=1)


class ModifiedSectionChunk(models.Model):
    section_embedding = models.ForeignKey(SectionEmbedding, on_delete=models.CASCADE)
    modified_section = models.ForeignKey(ModifiedSection, on_delete=models.CASCADE)
    chunk_number = models.IntegerField()
    chunk_text = models.TextField()


class Patent(models.Model):
    patent_ref = models.CharField(max_length=30)
    id = models.BigAutoField(primary_key=True)
    full_patent_ref = models.CharField(max_length=30, default="None")
    patent_country = models.CharField(max_length=10)
    patent_number = models.CharField(max_length=30)
    patent_type = models.CharField(max_length=30, default="Utility")
    patent_kind_code = models.CharField(max_length=3, default="")

    def __str__(self):
        return self.patent_ref


class PatentClaim(models.Model):
    id = models.BigAutoField(primary_key=True)
    patent = models.ForeignKey(Patent, on_delete=models.CASCADE)
    claim_number = models.IntegerField()
    claim_id = models.CharField(max_length=30)
    orig_claim_id = models.CharField(max_length=30, default="")
    related_claim = models.BooleanField(default=False)
    key_related_claim = models.BooleanField(default=False)
    text = models.TextField(default="", help_text="Text of Claim")

    def __str__(self):
        return self.claim_id


class ClaimElement(models.Model):
    id = models.BigAutoField(primary_key=True)
    claim = models.ForeignKey(PatentClaim, on_delete=models.CASCADE)
    element_id = models.CharField(max_length=30)
    element_number = models.IntegerField()
    text = models.TextField(default="")

    def __str__(self):
        return f"{self.claim.claim_id} Element {self.element_number}"


class ClaimRelatedSection(models.Model):
    claim = models.ForeignKey(PatentClaim, on_delete=models.CASCADE)
    related_sections = ArrayField(models.CharField(max_length=100))
    related_tables = ArrayField(models.CharField(max_length=100))
    related_figures = ArrayField(models.CharField(max_length=100))


class ModifiedClaim(models.Model):
    claim = models.ForeignKey(PatentClaim, on_delete=models.CASCADE)
    modification_type = models.ForeignKey(ModificationType, on_delete=models.CASCADE)
    modified_text = models.TextField()
    openai_tokens = models.IntegerField()

    def __str__(self):
        return f"{self.claim.claim_id}"


class ClaimEmbedding(models.Model):
    embed_type = models.ForeignKey(EmbeddingType, on_delete=models.CASCADE)
    source = models.ForeignKey(ModifiedClaim, on_delete=models.CASCADE)


class EmbeddingBaseModel(models.Model):
    embed_source = models.CharField(max_length=10, help_text="Indicates whether the embedding is for a claim or document")
    embed_id = models.BigIntegerField(help_text="ID in (Claim Embedding/Section Embeding) table that indicates the modified text and embedding type  ")
    source_id = models.BigIntegerField(help_text="ID to the document/claim xxForEmbedding) record that text to be used for embeddning (after modificaitons, before chunking)")
    orig_source_id = models.BigIntegerField(help_text="ID to the (doc/claim) record that has the original source text in the documents (before any modifications)")
    embed_type_name = models.CharField(max_length=200)
    mod_type_name = models.CharField(max_length=200)
    embed_type_shortname = models.CharField(max_length=20, null=True)
    chunk_number = models.IntegerField(default=1)
    total_chunks = models.IntegerField(default=1)

    class Meta:
        abstract = True


class Embedding1536(EmbeddingBaseModel):
    embedding_vector = VectorField(
        dimensions=1536,
        help_text="1536 element Vector embedding",
        null=True,
        blank=True
    )


class Embedding768(EmbeddingBaseModel):
    embedding_vector = VectorField(
        dimensions=768,
        help_text="768 element Vector embedding",
        null=True,
        blank=True
    )


class Embedding32(EmbeddingBaseModel):
    embedding_vector = VectorField(
        dimensions=32,
        help_text="32 element Vector embedding",
        null=True,
        blank=True
    )
