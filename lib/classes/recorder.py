from pathlib import Path
from abc import ABCMeta, abstractmethod
from uuid import uuid4

from py_pglite.sqlalchemy import SQLAlchemyPGliteManager # type: ignore[import-untyped]
from sqlalchemy import Engine, text

from lib.types.log import EventBase

# TODO: move this into global/ environment variables
EVENT_LOGGING = True

class RecorderBase(metaclass = ABCMeta):
    def __init__(self, *,  path: Path, run_id: int|None = None):
        self.path = path
        if run_id is None:
            pass
        else:
            self.run_id = uuid4().int


    @abstractmethod
    def log(self, event: EventBase) -> None:
        raise RuntimeError("must be subclassed!")


class JsonRecorder(RecorderBase):
    def __init__(self, path: Path, run_id: int|None = None):
        super().__init__(run_id=run_id, path=path)

    def log(self, event: EventBase):
        if EVENT_LOGGING:
            event.run_id = self.run_id
            with open(self.path, "a") as f:
                f.write(event.model_dump_json() + "\n")


class PostGresSQLRecorder(RecorderBase):

    def __init__(self, path: Path, run_id: int|None = None):
        super().__init__(run_id=run_id, path=path)

    def create_db(self, engine: Engine):
        create_sql = """
            CREATE TABLE runs (
                id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
            );

            CREATE TYPE memory_clear_reason AS ENUM (
                'too_many_out_of_bounds',
                'thought_loop'
            );

            CREATE TABLE thought_events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id INT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                datetime TIMESTAMPTZ NOT NULL,
                thought TEXT NOT NULL,
                target INT[]
            );

            CREATE TABLE memory_clear_events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id INT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                datetime TIMESTAMPTZ NOT NULL,
                reason memory_clear_reason NOT NULL
            );

            CREATE TABLE malformed_json_events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id INT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                datetime TIMESTAMPTZ NOT NULL,
                content TEXT NOT NULL
            );

            CREATE TABLE begin_simulation_events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id INT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                datetime TIMESTAMPTZ NOT NULL,
                params JSONB NOT NULL
            );
        """
        with engine.connect() as conn:
            conn.execute(text("SELECT version()")).scalar()

    def insert(self, event: EventBase):
        pass

class PGLiteSQLRecorder(PostGresSQLRecorder):
    
    def __init__(self, path: Path, run_id: int|None = None):
        super().__init__(run_id=run_id, path=path)

    def log(self, event: EventBase):
        if EVENT_LOGGING:
            event.run_id = self.run_id
            with SQLAlchemyPGliteManager() as db:
                engine: Engine = db.get_engine()

                if not self.path.exists():
                    self.create_db(engine)

                self.insert(event)
