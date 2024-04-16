from llama_index.core import PromptTemplate

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about world population statistics and details about a country. """

# This instruction string tells the engine what it needs to do with the Pandas data
instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

# This is templating what we want the prompt to look like to make it more human-readable for the user.
new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)