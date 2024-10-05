# Generated by Django 5.1 on 2024-10-01 20:42

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0012_embeddingtype_short_name"),
    ]

    operations = [
        migrations.AddField(
            model_name="embedding32",
            name="embed_type_shortname",
            field=models.CharField(max_length=20, null=True),
        ),
        migrations.AddField(
            model_name="embedding768",
            name="embed_type_shortname",
            field=models.CharField(max_length=20, null=True),
        ),
        migrations.AlterField(
            model_name="embedding32",
            name="orig_source_id",
            field=models.BigIntegerField(
                help_text="ID to the document/claim record that has the original source text in the documents (before any modifications)"
            ),
        ),
        migrations.AlterField(
            model_name="embedding32",
            name="source_id",
            field=models.BigIntegerField(
                help_text="ID to the document/claim xxForEmbedding) record that has text that used for embeddning"
            ),
        ),
    ]
