# Generated by Django 5.1 on 2024-11-02 21:22

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0023_claimchunkinfo_total_chunks"),
    ]

    operations = [
        migrations.RenameField(
            model_name="modifiedsectionchunk",
            old_name="modified_section",
            new_name="modified_item",
        ),
        migrations.CreateModel(
            name="ModifiedClainChunk",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("chunk_number", models.IntegerField()),
                ("chunk_text", models.TextField()),
                (
                    "chunk_info",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="polls.claimchunkinfo",
                    ),
                ),
                (
                    "modified_item",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="polls.modifiedclaim",
                    ),
                ),
            ],
        ),
    ]