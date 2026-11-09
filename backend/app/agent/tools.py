from langchain_core.tools import create_retriever_tool

from app.rag.retriever import get_retriever

SEARCH_PORTFOLIO_DESCRIPTION = (
    "Search Muhammad Waseem's resume and portfolio website. Use this for any "
    "question about his skills, tools, work experience, internships, education, "
    "projects, or contact details (email, phone, location, LinkedIn, GitHub). "
    "Always write the query in English, even if the user wrote in Urdu or "
    "Roman Urdu. Use topic keywords (for example 'skills and tools', "
    "'projects', 'contact email phone', 'internship experience') and do not "
    "put the person's name in the query."
)


def get_tools() -> list:
    search_portfolio = create_retriever_tool(
        get_retriever(),
        name="search_portfolio",
        description=SEARCH_PORTFOLIO_DESCRIPTION,
    )
    return [search_portfolio]