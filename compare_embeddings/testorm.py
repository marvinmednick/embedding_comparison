import os 
import django 
from django.utils import timezone

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'compare_embeddings.settings')
django.setup()


from polls.models import Question


print(f"Questions - {Question.objects.all()}")
q = Question(question_text="What's new dude?", pub_date=timezone.now())
q.save()


print(f"Questions - {Question.objects.all()}")
