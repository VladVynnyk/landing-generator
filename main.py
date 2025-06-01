import os

from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents import director_agent, content_creator_agent, code_agent, relevancy_checker_agent, image_generator_agent, image_idea_agent 

# Goal: generate html files (landings) with content (text, images)
# Steps to make:
# 1. Create a "plan" of html file. User needs to add important details about it. How many sections should be inside. Which content should be, CTA's, etc.
# A goal of landing. What we want to achieve? 
# 2. Creation of content.
# 3. Validation of content. Is it good or not. (Human-in-the-loop)
# 4. Creation of html landing.
# 5. CSS styles generation.
# 6. Validation of html landing. (Human-in-the-loop)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#company_service = input("What do you or your company does?: ")
#name_of_landing = input("Type a name of landing: ")
#country_of_target_audience = input("Type a country of target audience: ")
#goal_of_landing = input("Type a goal of landing: ")
#amount_of_sections = input("How many sections landing should have?: ")
#ideas_exists = input("Do you have some ideas for sections? (Yes/No): ")
#if ideas_exists == "yes".lower():
#    ideas_for_sections = input("Tell me everything what you want to see: ")

# if yes 
# if no we will go to CTA
#include_CTA = input("CTA should be included? (Yes/No): ")
#if include_CTA == "yes".lower():
#    CTA_ideas = input("Do you have some CTA ideas?")

#include_photos = input("Photos should be included? (Yes/No)")
#if include_photos == "yes".lower():
#    amount_of_photos = input("How many photos do you want?")
#    ideas_for_photos_exists = input("Do you have some ideas for photos? (Yes/No)")
#    if ideas_for_photos_exists == "yes".lower():
#        ideas_for_photos = input("Describe images you want to make for landing: ")
# if yes while loop
# if no we will go on to the 

class LandingState(TypedDict, total=False):
    user_input: dict
    structured_agent_prompts: dict
    content_output: dict
    image_prompts_output: dict
    images_output: dict
    full_html_page: str
    qa_output: dict

builder = StateGraph(state_schema=LandingState)

builder.add_node("DirectorAgent", director_agent)
builder.add_node("ContentCreatorAgent", content_creator_agent)
builder.add_node("ImageIdeaAgent", image_idea_agent)
builder.add_node("ImageGeneratorAgent", image_generator_agent)
builder.add_node("CodeAgent", code_agent)
builder.add_node("RelevancyCheckerAgent", relevancy_checker_agent)

builder.set_entry_point("DirectorAgent")

builder.add_edge("DirectorAgent", "ContentCreatorAgent")
builder.add_edge("ContentCreatorAgent", "ImageIdeaAgent")
# builder.add_edge("ImageIdeaAgent", "ImageGeneratorAgent")
# builder.add_edge("ImageGeneratorAgent", "CodeAgent")
builder.add_edge("ImageIdeaAgent", "CodeAgent")
# builder.add_edge("CodeAgent", "RelevancyCheckerAgent")
# builder.add_edge("RelevancyCheckerAgent", END)
builder.add_edge("CodeAgent", END)


graph = builder.compile()

user_input = {
    "company_service": "IT consulting",
    "name_of_landing": "IT consulting",
    "language_of_landing": "Українська",
    "country_of_target_audience": "Ukraine",
    "goal_of_landing": "To sell consultations to small and big busineses",
    "amount_of_sections": "5",
    # "ideas_for_sections": "Features, Testimonials, Pricing",
    "ideas_for_sections": "",
    "include_CTA": "Yes",
    "CTA_ideas": "Buy first consultation with discount",
    "include_photos": "Yes",
    "amount_of_photos": "3",
    "ideas_for_photos": "",
    "user_style": "modern"
}

result = graph.invoke({"user_input": user_input})

with open("it-consulting.html", "w", encoding="utf-8")as f:
    f.write(result["full_html_page"])

print("Landing saved to landing.html")
