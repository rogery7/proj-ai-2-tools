import subprocess
import requests
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.tools.shell.tool import ShellTool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from bs4 import BeautifulSoup

CACHE_FILE = "jira_cli_readme_cache.txt"

def fetch_jira_cli_readme():
    """
    Fetch GitHub repository README content and cache it.

    Returns:
    - str: README content.
    """
    github_url = "https://github.com/ankitpokhrel/jira-cli"

    try:
        # Check if README content is cached
        try:
            with open(CACHE_FILE, "r") as file:
                return file.read()
        except FileNotFoundError:
            pass

        # Fetch README content
        response = requests.get(f"{github_url}/blob/master/README.md")
        response.raise_for_status()

        # Cache README content
        with open(CACHE_FILE, "w") as file:
            file.write(response.text)

        return response.text
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"Error: {e}"
    
def generate_jira_cli_commands_from_readme(readme_content):
    """
    Generate Jira-CLI commands from README content.

    Args:
    - readme_content (str): README content.

    Returns:
    - list of str: Generated Jira-CLI commands.
    """
    try:
        # Parse HTML content
        soup = BeautifulSoup(readme_content, 'html.parser')

        # Find README content
        readme_content = soup.find('article', class_='markdown-body')

        if readme_content:
            # Extract text from README content
            readme_text = readme_content.get_text()

            # Split text into lines
            lines = readme_text.split('\n')

            # Filter lines containing command examples
            command_lines = [line.strip() for line in lines if line.strip().startswith("$ jira")]

            return command_lines
        else:
            return ["Documentation not found in README"]

    except Exception as e:
        return [f"Error: {e}"]

@tool
def run_command_in_directory(directory, command):
    """
    Run a command in a directory and return the stdout and stderr
    """
    result = subprocess.run(
        command,
        cwd=directory,
        shell=True,
        text=True,
        capture_output=True
    )
    return result.stdout, result.stderr

@tool
def get_issue_from_git_branch(branch_name):
    """
    Get the issue number from a git branch name
    """
    tokens = branch_name.split('-')
    first_two_tokens = tokens[:2]
    return '-'.join(first_two_tokens)

@tool
def jira_cli_commands_tool():
    """
    Jira-CLI commands tool.

    Returns:
    - str: Tool output.
    """
    # Fetch README content
    readme_content = fetch_jira_cli_readme()

    # Generate commands from README content
    generated_commands = generate_jira_cli_commands_from_readme(readme_content)

    # Format output
    output = "\n".join(generated_commands)
    return output

@tool
def get_jira_cli_update_description_command(issue_key, new_description):
    """
    Update Jira issue description.

    Args:
    - issue_key (str): Jira issue key.
    - new_description (str): New description.

    Returns:
    - str: Update status.
    """

    # the jira_cli_commands_tool is unable to form this command for some reason
    return f"jira issue edit {issue_key} -b'{new_description}' --no-input"

@tool
def get_project_directory():
    """
    Get the project directory
    """
    return "YOUR_PROJECT_DIRECTORY"

tools = [
    run_command_in_directory,
    get_issue_from_git_branch,
    jira_cli_commands_tool,
    get_project_directory,
    get_jira_cli_update_description_command
]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an agent that keeps Git and Jira in sync. Do not add or delete anything, only update. Only output the final result.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="git_jira_agent"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "git_jira_agent": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    user_prompt = input("Prompt: ")
    list(agent_executor.stream({"input": user_prompt}))