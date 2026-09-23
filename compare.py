import os
import time
import requests

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# LOAD API KEYS
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------
# MODELS WE WANT THE ROUTERS TO CHOOSE BETWEEN
# --------------------------------------------------

MODELS = {
    "openai": "openai/gpt-5.6-luna",
    "gemini": "google/gemini-2.5-flash"
}


# --------------------------------------------------
# OPENAI ROUTER
# --------------------------------------------------

def openai_router(user_prompt):

    router_prompt = f"""
You are an AI model router.

Your job is ONLY to decide which model should handle
the user's request.

Available models:

1. openai
   Model: {MODELS["openai"]}
   Description:
   Good for reasoning, coding, complex problem solving,
   and tasks requiring detailed logical analysis.

2. gemini
   Model: {MODELS["gemini"]}
   Description:
   Good for general questions, summarization,
   and tasks where strong general-purpose performance is sufficient.

Choose the model that is most appropriate for the user's request.

IMPORTANT:
Return ONLY one word:

openai

OR

gemini

Do not explain your answer.

User request:
{user_prompt}
"""

    start_time = time.perf_counter()

    response = openai_client.responses.create(
        model="gpt-5.6-luna",
        input=router_prompt
    )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    decision = response.output_text.strip().lower()

    # Make sure OpenAI returned a valid choice
    if decision not in ["openai", "gemini"]:
        decision = "invalid"

    # Usage information
    usage = response.usage

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens

    # GPT-5.6 Luna pricing:
    # $0.20 / 1M input tokens
    # $1.20 / 1M output tokens

    cost = (
        (input_tokens / 1_000_000) * 0.20
        +
        (output_tokens / 1_000_000) * 1.20
    )

    return {
        "decision": decision,
        "time": elapsed_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost
    }


# --------------------------------------------------
# JEV ROUTER
# --------------------------------------------------

def jev_router(user_prompt):

    url = "https://openrouter.ai/api/alpha/decisions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "typesafe/jev-1.13",

        "state": user_prompt,

        "questions": {
            "model": {
                "type": "choice",

                "instructions": (
                    "Which AI model should handle this user request?"
                ),

                "criteria": {
                    "openai": (
                        f"Choose this for reasoning-heavy, coding, "
                        f"complex problem solving, and detailed logical tasks. "
                        f"Model: {MODELS['openai']}"
                    ),

                    "gemini": (
                        f"Choose this for general questions, summarization, "
                        f"and general-purpose tasks. "
                        f"Model: {MODELS['gemini']}"
                    )
                }
            }
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    response.raise_for_status()

    data = response.json()

    decision = data["answers"]["model"]["choice"]

    # Jev usage
    usage = data.get("usage", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Jev pricing:
    # $0.042 / 1M input tokens
    # $0 / 1M output tokens

    cost = (
        (input_tokens / 1_000_000) * 0.042
    )

    return {
        "decision": decision,
        "time": elapsed_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost
    }


# --------------------------------------------------
# GET 5 PROMPTS
# --------------------------------------------------

prompts = []

print("\n======================================")
print(" ENTER 5 PROMPTS")
print("======================================\n")

for i in range(1, 6):

    prompt = input(f"Prompt {i}: ")

    prompts.append(prompt)


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

results = []

print("\n\n======================================")
print(" STARTING COMPARISON")
print("======================================\n")


for i, prompt in enumerate(prompts, start=1):

    print("--------------------------------------")
    print(f"PROMPT {i}")
    print("--------------------------------------")

    print(f"User: {prompt}\n")

    # ----------------------------------------------
    # OPENAI
    # ----------------------------------------------

    print("Running OpenAI Router...")

    openai_result = openai_router(prompt)

    print(
        f"OpenAI decision : "
        f"{openai_result['decision']}"
    )

    print(
        f"OpenAI time     : "
        f"{openai_result['time']:.4f} seconds"
    )

    print(
        f"OpenAI tokens   : "
        f"{openai_result['input_tokens']} input + "
        f"{openai_result['output_tokens']} output"
    )

    print(
        f"OpenAI cost     : "
        f"${openai_result['cost']:.8f}"
    )


    # ----------------------------------------------
    # JEV
    # ----------------------------------------------

    print("\nRunning Jev Router...")

    jev_result = jev_router(prompt)

    print(
        f"Jev decision    : "
        f"{jev_result['decision']}"
    )

    print(
        f"Jev time        : "
        f"{jev_result['time']:.4f} seconds"
    )

    print(
        f"Jev tokens      : "
        f"{jev_result['input_tokens']} input + "
        f"{jev_result['output_tokens']} output"
    )

    print(
        f"Jev cost        : "
        f"${jev_result['cost']:.8f}"
    )


    # ----------------------------------------------
    # AGREEMENT
    # ----------------------------------------------

    if openai_result["decision"] == jev_result["decision"]:
        agreement = "YES"
    else:
        agreement = "NO"


    print(
        f"\nSame decision?  : {agreement}"
    )


    # Save results

    results.append({
        "prompt": prompt,

        "openai_decision": openai_result["decision"],
        "openai_time": openai_result["time"],
        "openai_cost": openai_result["cost"],

        "jev_decision": jev_result["decision"],
        "jev_time": jev_result["time"],
        "jev_cost": jev_result["cost"],

        "agreement": agreement
    })


# --------------------------------------------------
# TOTALS
# --------------------------------------------------

openai_total_time = sum(
    r["openai_time"] for r in results
)

jev_total_time = sum(
    r["jev_time"] for r in results
)

openai_total_cost = sum(
    r["openai_cost"] for r in results
)

jev_total_cost = sum(
    r["jev_cost"] for r in results
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n\n======================================")
print(" FINAL COMPARISON")
print("======================================\n")

print(
    f"OpenAI total time : "
    f"{openai_total_time:.4f} seconds"
)

print(
    f"Jev total time    : "
    f"{jev_total_time:.4f} seconds"
)

print()

print(
    f"OpenAI total cost : "
    f"${openai_total_cost:.8f}"
)

print(
    f"Jev total cost    : "
    f"${jev_total_cost:.8f}"
)

print()

agreement_count = sum(
    1 for r in results
    if r["agreement"] == "YES"
)

print(
    f"Same decisions    : "
    f"{agreement_count}/5"
)

print("\n======================================")
print(" PROMPT-BY-PROMPT RESULTS")
print("======================================\n")

for i, r in enumerate(results, start=1):

    print(f"Prompt {i}")
    print(f"  OpenAI -> {r['openai_decision']}")
    print(f"  Jev    -> {r['jev_decision']}")
    print(f"  Same   -> {r['agreement']}")
    print()