# Generated by Django 5.1 on 2024-10-27 18:27

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0018_embedding1536"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="SectionForEmbedding",
            new_name="ModifiedDocSection",
        ),
    ]