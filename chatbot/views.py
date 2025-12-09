from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag.utils import ask_llm
from django.http import HttpResponse
from django.shortcuts import render

class ChatAPIView(APIView):
    def post(self, request):
        question = request.data.get("question", "")
        if not question:
            return Response({"error": "Pertanyaan diperlukan"}, status=400)

        answer = ask_llm(question)
        return Response({"answer": answer}, status=200)
    
def index(request):
        return HttpResponse("Chatbot KUA aktif. Gunakan endpoint /chat/")

def landing_page(request):
    return render(request, "chatbot/landing.html")

def chat_ui(request):
    return render(request, "chatbot/chat.html")



