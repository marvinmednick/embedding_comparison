# Generated by Django 5.1.2 on 2024-11-06 22:39

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0031_remove_modifiedclaim_openai_tokens_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="Tag",
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
                ("name", models.CharField(max_length=50, unique=True)),
            ],
        ),
        migrations.AddField(
            model_name="embedding1536",
            name="tags",
            field=models.ManyToManyField(related_name="%(class)s", to="polls.tag"),
        ),
        migrations.AddField(
            model_name="embedding32",
            name="tags",
            field=models.ManyToManyField(related_name="%(class)s", to="polls.tag"),
        ),
        migrations.AddField(
            model_name="embedding768",
            name="tags",
            field=models.ManyToManyField(related_name="%(class)s", to="polls.tag"),
        ),
    ]
