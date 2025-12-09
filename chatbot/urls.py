from django.urls import path
from .views import ChatAPIView, index, chat_ui
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('chat/', views.chat_ui, name='chat_ui'),
    path("api/chat/", ChatAPIView.as_view()),
    path("ping/", index),  # opsional untuk status
]
