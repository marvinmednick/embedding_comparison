# Generated by Django 5.1 on 2024-11-02 21:25

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        (
            "polls",
            "0024_rename_modified_section_modifiedsectionchunk_modified_item_and_more",
        ),
    ]

    operations = [
        migrations.RenameField(
            model_name="modifiedsection",
            old_name="section",
            new_name="item",
        ),
    ]
