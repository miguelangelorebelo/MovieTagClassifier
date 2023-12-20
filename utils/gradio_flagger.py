from datetime import datetime
from pathlib import Path
from typing import Any

from gradio.components import IOComponent
from gradio.flagging import FlaggingCallback
from loguru import logger
from sqlalchemy import Table
from sqlalchemy.dialects.mysql import insert

from database.database import Base, engine, db_name, table_name

engine.execute(f"USE {db_name}")
Base.metadata.bind = engine
conti_table = Table(table_name, Base.metadata, autoload=True, autoload_with=engine)


class CustomFlagger(FlaggingCallback):

    def __init__(self):
        self.components = None
        self.flagging_dir = None
        pass

    def setup(
            self,
            components: list[IOComponent],
            flagging_dir: str | Path
    ):
        self.components = components
        self.flagging_dir = flagging_dir

    def flag(
            self,
            flag_data: list[Any],
            flag_option: str = "",
            username: str | None = None,
    ) -> int:
        data = {
            'flag': flag_option,
            'prompt': flag_data[0],
            'model': flag_data[1],
            'response': flag_data[2],
            'elapsed_seconds': flag_data[3],
            'date_time': datetime.now()
        }

        stmt = insert(conti_table).values(**data)
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.close()
            logger.info('Flagged content inserted into db')

        return len(self.components)
