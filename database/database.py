from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base

from configs.db_config import username, password, ip, port

db_name = "enhesa_db"
table_name = "movie_predictions"

engine = create_engine(f"mysql+pymysql://{username}:{password}@{ip}:{port}")

Base = declarative_base()
