# Generated by Django 5.1 on 2024-10-03 04:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0014_popluate_short_name"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="filename",
            field=models.TextField(default="NA", help_text="Name of Document"),
            preserve_default=False,
        ),
    ]
