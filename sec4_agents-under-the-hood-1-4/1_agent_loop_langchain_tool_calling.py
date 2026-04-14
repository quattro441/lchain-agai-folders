from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b "

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentage.get(discount_tier, 0)

    return round(price * ((1 - discount) / 100), 2)

def run_agent(question: str):
    pass


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying the discount?")