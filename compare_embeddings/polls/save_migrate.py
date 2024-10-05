# Generated by Django 5.1 on 2024-08-22 19:44

from django.db import migrations, models


def create_populate_id(model_name):
    def populate_id(apps, schema_editor):
        # Dynamically get the model using the model name
        Model = apps.get_model('polls', model_name)
        for index, obj in enumerate(Model.objects.all(), start=1):
            print(model_name, index)
            obj.id = index
            obj.save()
    return populate_id


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0018_embedding32_source_id_embedding768_orig_source_id_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="claimelement",
            name="id",
            field=models.BigIntegerField(null=True),
        ),
        migrations.AddField(
            model_name="patent",
            name="id",
            field=models.BigIntegerField(null=True),
        ),
        migrations.AddField(
            model_name="patentclaim",
            name="id",
            field=models.BigIntegerField(null=True),
        ),
        migrations.AlterField(
            model_name="embedding768",
            name="orig_source_id",
            field=models.BigIntegerField(
                help_text="ID to the document/claim record that has the original source text in the documents (before any modifications)"
            ),
        ),
        migrations.AlterField(
            model_name="embedding768",
            name="source_id",
            field=models.BigIntegerField(
                help_text="ID to the document/claim xxForEmbedding) record that has text that used for embeddning"
            ),
        ),
        migrations.RunPython(create_populate_id('claimelement')),
        migrations.RunPython(create_populate_id('patent')),
        migrations.RunPython(create_populate_id('patentclaim')),
    ]
