# Generated by Django 5.1 on 2024-08-22 20:54

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("polls", "0002_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="patentclaim",
            name="claim_text",
        ),
        migrations.AddField(
            model_name="patentclaim",
            name="text",
            field=models.TextField(default="", help_text="Text of Claim"),
        ),
    ]
