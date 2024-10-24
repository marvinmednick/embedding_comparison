# Generated by Django 5.1 on 2024-08-23 15:50

import datetime
import django.utils.timezone
from django.db import migrations, models

def populate_mod_type768(apps, schema_editor):
    ThisModel = apps.get_model('polls', 'embedding768')
    DocModel = apps.get_model('polls', 'SectionEmbedding')
    ClaimModel = apps.get_model('polls', 'ClaimEmbedding')

    for index, obj in enumerate(ThisModel.objects.all(), start=1):
        if index % 100 == 0:
            print(f"Processing mod type 768 {index}")
        embed_id = obj.embed_id
        if obj.embed_source == 'document':
            embedding_model = DocModel
        elif obj.embed_source == 'claim':
            embedding_model = ClaimModel
        else:
            raise ValueError(f"'{obj.embed_source}' is not a valid source type.")

        embed_obj = embedding_model.objects.get(id=embed_id)
        obj.mod_type_name = embed_obj.source.modification_type.name
        obj.save()


def populate_mod_type32(apps, schema_editor):
    ThisModel = apps.get_model('polls', 'embedding32')
    DocModel = apps.get_model('polls', 'SectionEmbedding')
    ClaimModel = apps.get_model('polls', 'ClaimEmbedding')

    for index, obj in enumerate(ThisModel.objects.all(), start=1):
        if index % 100 == 0:
            print(f"Processing mod type 32 {index}")
        embed_id = obj.embed_id
        if obj.embed_source == 'document':
            embedding_model = DocModel
        elif obj.embed_source == 'claim':
            embedding_model = ClaimModel
        else:
            raise ValueError(f"'{obj.embed_source}' is not a valid source type.")

        embed_obj = embedding_model.objects.get(id=embed_id)
        obj.mod_type_name = embed_obj.source.modification_type.name
        obj.save()


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0004_rename_element_text_claimelement_text"),
    ]

    operations = [
        migrations.AddField(
            model_name="embedding32",
            name="mod_type_name",
            field=models.CharField(default="Unknown", max_length=20),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="embedding768",
            name="mod_type_name",
            field=models.CharField(default="Unknown", max_length=20),
            preserve_default=False,
        ),
        migrations.RunPython(populate_mod_type768),
        migrations.RunPython(populate_mod_type32),
    ]
