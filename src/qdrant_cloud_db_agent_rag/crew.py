import os

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from pydantic import BaseModel
from qdrant_cloud_db_agent_rag.tools.qdrant_vector_search import QdrantVectorSearchTool

load_dotenv()


@CrewBase
class QdrantCloudDbAgentRag:
    """QdrantCloudDbAgentRag crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    vector_search_tool = QdrantVectorSearchTool(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name="contracts23",
    )

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],
            verbose=True,
            tools=[self.vector_search_tool],
        )

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["reporter"],
            verbose=True,
            tools=[self.vector_search_tool],
        )

    @task
    def rag_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_task"],
        )

    @task
    def reporter_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporter_task"],
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the QdrantCloudDbAgentRag crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
