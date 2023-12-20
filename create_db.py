import time

from loguru import logger

from database.model import MovieModel
from database.database import Base, engine, db_name

if __name__ == "__main__":
    # Create db
    engine.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    engine.execute(f"USE {db_name}")

    # Create tables
    Base.metadata.bind = engine
    Base.metadata.create_all(engine)
    logger.info("Table successfully created")
    time.sleep(5)
