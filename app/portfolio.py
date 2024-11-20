import pandas as pd
import chromadb
import uuid
class Portfolio:
    def __init__(self, file_path="app/resource/CSV_Data_2024_11_19 13_49.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["TechStack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])

    def calculate_matching_percentage(self, required_skills):
        portfolio_skills = self.get_skills()
        matched_skills = list(set(required_skills) & set(portfolio_skills))
        match_percentage = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0

        return match_percentage, matched_skills

    def get_skills(self):
        return self.data["TechStack"].dropna().tolist()