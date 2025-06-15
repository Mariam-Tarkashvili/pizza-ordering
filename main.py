import os
import json
import openai
import pandas as pd
from dotenv import load_dotenv
import re

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama3-8b-8192"


df = pd.read_csv("dataset.csv")
named_pizzas = df[df["category"] == "named_pizza"]
sizes = df[df["category"] == "size"]["name"].tolist()
crusts = df[df["category"] == "crust"]["name"].tolist()
toppings = df[df["category"] == "topping"]["name"].tolist()
sauces = df[df["category"] == "sauce"]["name"].tolist()
named_pizza_names = named_pizzas["name"].str.lower().tolist()


def order_pizza(size, crust, toppings, sauces):
    base_price = {"small": 5, "medium": 7, "large": 9, "extra large": 11}
    price = base_price.get(size.lower(), 7)
    price += 0.75 * len(toppings) + 0.5 * len(sauces)
    return {
        "status": "success",
        "order": {
            "size": size,
            "crust": crust,
            "toppings": toppings,
            "sauces": sauces,
            "price": round(price, 2)
        },
        "eta": "20 minutes"
    }


order_tool = {
    "type": "function",
    "function": {
        "name": "order_pizza",
        "description": "Place a pizza order with size, crust, toppings, and sauces",
        "parameters": {
            "type": "object",
            "required": ["size", "crust", "toppings", "sauces"],
            "properties": {
                "size": {"type": "string"},
                "crust": {"type": "string"},
                "toppings": {"type": "array", "items": {"type": "string"}},
                "sauces": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
}


def main():
    print("👋 Welcome to Tarka's PizzaHut")
    raw_input = input("Your name: ")
    match = re.search(r"(?:my name is|i am|i'm)\s+(.+)", raw_input, re.IGNORECASE)
    if match:
        name = match.group(1).strip().title()
    else:
        name = raw_input.strip().title()
    print(f"Hi {name}! Let's start your order.")

    messages = [
        {
            "role": "system",
            "content":
                "You are a helpful pizza-ordering assistant.\n"
                "Here is the available menu:\n"
                f"Sizes: {', '.join(sizes)}\n"
                f"Crusts: {', '.join(crusts)}\n"
                f"Toppings: {', '.join(toppings)}\n"
                f"Sauces: {', '.join(sauces)}\n"
                f"Named pizzas: {', '.join(named_pizza_names)}\n\n"
                "ONLY accept orders with these exact items. "
                "If the user asks for anything else, politely explain it's not available and ask them to choose from the list."
                "Name of pizza place is Tarka's PizzaHut"
        }
    ]

    orders = []
    total = 0.0

    while True:
        user_msg = input(f"\n{name}: ")
        messages.append({"role": "user", "content": user_msg})

        while True:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                tools=[order_tool],
                tool_choice="auto"
            )
            reply = response["choices"][0]["message"]
            messages.append({"role": "assistant", "content": reply.get("content", "")})
            if reply.get("content"):
                print(f"\n🤖 AI: {reply['content']}")

            if "tool_calls" in reply:
                for tool_call in reply["tool_calls"]:
                    args = json.loads(tool_call["function"]["arguments"])
                    result = order_pizza(**args)
                    print(f"\\n✅ Order placed! ETA: {result['eta']}")
                    print(f"💰 Price: ${result['order']['price']:.2f}")
                    orders.append(result["order"])
                    total += result["order"]["price"]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "order_pizza",
                        "content": json.dumps(result)
                    })
                break  # ✅ Finished ordering current pizza

            # if not tool yet, get a user reply to AI's question
            user_msg = input(f"\n{name}: ")
            messages.append({"role": "user", "content": user_msg})

        cont = input("\\nWould you like to order another pizza? (yes/no): ").strip().lower()
        if cont != "yes":
            break

    print("\\n🧾 Final Order Summary:")
    for i, o in enumerate(orders, 1):
        print(f"Pizza {i}: {o['size']} {o['crust']} crust")
        print(f" Toppings: {', '.join(o['toppings'])}")
        print(f" Sauces: {', '.join(o['sauces'])}")
        print(f" Price: ${o['price']:.2f}\\n")
    print(f"💳 Total Bill: ${total:.2f}")
    print("🍕 Your pizzas will be ready soon. Thank you!")


if __name__ == "__main__":
    main()
