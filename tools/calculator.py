# Simple function with basic types
from tools.toolset import toolset


@toolset.add()
def hello(name: str) -> str:
    """
    A simple function that greets the user.
    :param name: the name of the user. example: "World"
    :return: the greeting message. example: "Hello, World!"
    """
    return f"Hello, {name}!"

# Function with multiple parameters
@toolset.add()
def add(a: int, b: int) -> int:
    """
    Add two integers together.
    :param a: the first integer
    :param b: the second integer
    :return: the sum of a and b
    """
    return a + b

# Function with custom path
@toolset.add("multiply")
def multiply_numbers(x: float, y: float) -> float:
    """
    Multiply two floating-point numbers.
    :param x: the first number
    :param y: the second number
    :return: the product of x and y
    """
    return x * y

