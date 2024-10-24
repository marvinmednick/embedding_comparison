# Generated by Django 5.1 on 2024-08-24 21:36

from django.db import migrations

import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4-turbo")


def calc_tokens(apps, schema_editor):
    Model = apps.get_model('polls', 'claimforembedding')
    calculate_openai_tokens(Model, 'claims')
    Model = apps.get_model('polls', 'sectionforembedding')
    calculate_openai_tokens(Model, 'sections')


def calculate_openai_tokens(Model, name):

    for index, obj in enumerate(Model.objects.all(), start=1):
        obj.openai_tokens = len(encoding.encode(obj.modified_text))
        print(name, index, obj.openai_tokens)
        obj.save()


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0007_claimforembedding_openai_tokens_and_more"),
    ]

    operations = [migrations.RunPython(calc_tokens)]
