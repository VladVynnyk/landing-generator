from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
import os 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
llm_4o = ChatOpenAI(model="gpt-4o")

parser = JsonOutputParser()

FORCE_JSON_SUFFIX = """
    You MUST return ONLY valid JSON. Do not add any text before or after.
    """

def director_agent(state):
    user_input = state["user_input"]

    prompt = f"""
    You are a Project Director for an AI landing page generator.

    Given the user input, generate well-structured and optimized prompts 
    for the following agents:
    - ContentCreatorAgent
    - ImageIdeaAgent
    - ImageGeneratorAgent
    - CodeAgent
    - RelevancyAndEfficiencyCheckerAgent

    User input:
    {json.dumps(user_input, indent=2)}

    Return a JSON object like:

    {{
      "ContentCreatorAgent_prompt": "...",
      "ImageIdeaAgent_prompt": "...",
      "ImageGeneratorAgent_prompt": "...",
      "CodeAgent_prompt": "...",
      "RelevancyAndEfficiencyCheckerAgent_prompt": "..."
    }}
    """
    
    print("[DirectorAgent] started")
    response = llm.invoke([HumanMessage(content=prompt)])
    structured_agent_prompts = parser.invoke(response)
    
    print("[DirectorAgent] parsed result")
    print(structured_agent_prompts)


    # Add suffix to each agent prompt that needs JSON:
    structured_agent_prompts["ContentCreatorAgent_prompt"] += "\n" + FORCE_JSON_SUFFIX
    structured_agent_prompts["ImageIdeaAgent_prompt"] += "\n" + FORCE_JSON_SUFFIX
    structured_agent_prompts["ImageGeneratorAgent_prompt"] += "\n" + FORCE_JSON_SUFFIX
    structured_agent_prompts["RelevancyAndEfficiencyCheckerAgent_prompt"] += "\n" + FORCE_JSON_SUFFIX

    return { "structured_agent_prompts": structured_agent_prompts }

def content_creator_agent(state):
    addition_prompt = """
    JSON structure should be the next: 
    {{
        "sections": "...",
        "CTA": "...",
        "images": "...",
        "user_style": "..."
    }}
    """
    content_creator_prompt = state["structured_agent_prompts"]["ContentCreatorAgent_prompt"] + "\n" + addition_prompt + "\n" + FORCE_JSON_SUFFIX

    print("[ContentCreatorAgent] parsed result")
    response = llm.invoke([HumanMessage(content=content_creator_prompt)])
    content_output = parser.invoke(response)

    print("[ContentCreatorAgent] parsed result")
    print(content_output)
    return { "content_output": content_output }

def image_idea_agent(state):
    image_idea_prompt = state["structured_agent_prompts"]["ImageIdeaAgent_prompt"]

    print("[ImageIdeaAgent] started")
    response = llm.invoke([HumanMessage(content=image_idea_prompt)])
    print("[ImageIdeaAgent] parsed result")
    image_prompts_output = parser.invoke(response)
    print(image_prompts_output)

    return { "image_prompts_output": image_prompts_output }

def image_generator_agent(state):
    image_prompts = state["image_prompts_output"]["image_prompts"]
    #image_prompts = state["image_prompts_output"]["imageConcepts"]

    print("[ImageGeneratorAgent] started")
    print("Generating image URLs...")

    generated_images = []
    for img in image_prompts:
        generated_images.append({
            "section": img["section"],
            "url": f"https://dummyimage.com/1024x768/cccccc/000000&text={img['section'].replace(' ', '+')}"
        })

    print("[ImageGeneratorAgent] generated images")
    print(generated_images)

    return { "images_output": { "images": generated_images } }

def code_agent(state):
    print("STATE: ", state)
    code_agent_prompt = state["structured_agent_prompts"]["CodeAgent_prompt"] + "\n" + "instead of pasting images make placeholders"+ "\n" + "You MUST return ONLY valid HTML and CSS. Do not add any text before or after."

    sections = state["content_output"]["sections"]
    CTA = state["content_output"]["CTA"]
    user_style = state["user_input"]["user_style"]

    code_agent_prompt = f"""
        You are an expert front-end developer.

        Create a complete responsive HTML5 landing page based on this content:

        Sections:
        {json.dumps(sections, indent=2)}

        CTA:
        {CTA}

        Style:
        {user_style}

        Requirements:
        - Responsive (mobile-first)
        - Semantic HTML5
        - Inline CSS or external stylesheet
        - Insert placeholders for images
        - Accessibility best practices

        You MUST return ONLY valid HTML and CSS. Do not add any text before or after.
        """


    # filled_prompt = code_agent_prompt.format(
    #     sections=json.dumps(sections, indent=2),
    #     CTA=CTA,
    #     # images=json.dumps(images),
    #     user_style=user_style
    # )

    print("Filled prompt: ", code_agent_prompt)

    print("[CodeAgent] started")
    #response = llm_4o.invoke([HumanMessage(content=filled_prompt)])
    response = llm_4o.invoke([HumanMessage(content=code_agent_prompt)])
    full_html_page = response.content
    
    print("[CodeAgent] raw HTML response")
    print(response.content)

    return { "full_html_page": full_html_page }

def relevancy_checker_agent(state):
    checker_prompt = state["structured_agent_prompts"]["RelevancyAndEfficiencyCheckerAgent_prompt"]

    html = state["full_html_page"]
    goal = state["user_input"]["goal_of_landing"]
    country = state["user_input"]["country_of_target_audience"]

    filled_prompt = checker_prompt.format(
        full_html_page=html,
        goal_of_landing=goal,
        country_of_target_audience=country
    )

    print("[RelevancyCheckerAgent] started")
    response = llm.invoke([HumanMessage(content=filled_prompt)])
    checker_output = parser.invoke(response)

    print("[RelevancyCheckerAgent] parsed result")
    print(checker_output)

    return { "qa_output": checker_output }

