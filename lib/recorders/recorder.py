from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import Any
from uuid import uuid4

import psycopg2

from lib.types.log import EventBase, ThoughtEvent, MemoryClearEvent, MalformedJSONEvent, BeginSimulationEvent

# TODO: move this into global/ environment variables
EVENT_LOGGING = True

class RecorderBase(metaclass = ABCMeta):
    is_open = False
    def __init__(self, run_id: int|None = None):
        if run_id is None:
            raise RuntimeError
            self.run_id = uuid4().int
        else:
            self.run_id = run_id

    @abstractmethod
    def __enter__(self) -> Any:
        self.__class__.is_open = True

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        self.__class__.is_open = False
        

    @abstractmethod
    def log(self, event: EventBase) -> None:
        raise RuntimeError("must be subclassed!")


class JsonRecorder(RecorderBase):
    def __init__(self, path: Path, run_id: int|None = None):
        self.path = path
        super().__init__(run_id)

    def __enter__(self):
        if EVENT_LOGGING:
            if not self.path.exists():
                self.path.touch()
            self.file = open(self.path, "a")
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "file"):
            self.file.close()
        super().__exit__(exc_type, exc_value, traceback)

    def log(self, event: EventBase):
        event.run_id = self.run_id
        self.file.write(event.model_dump_json() + "\n")

    

class PostGresSQLRecorder(RecorderBase):

    def __init__(self, run_id: int|None = None, path=None):
        super().__init__(run_id)

    def __enter__(self):
        if EVENT_LOGGING:
            self.connection = psycopg2.connect(
                database="postgres",
                user="postgres",
                password="password",
                host="localhost",
                port= "5432"
            )

            self.connection.autocommit = True
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()
        return super().__exit__(exc_type, exc_value, traceback)

    def execute(self, sql: str, params: dict|None = None):
        with self.connection.cursor() as cur:
            cur.execute(sql, params)
    
    def create_db(self):
        create_sql = """
        CREATE TYPE memory_clear_reason AS ENUM (
            'too_many_out_of_bounds',
            'thought_loop'
        );

        CREATE TABLE runs (
            id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            context_size INT NOT NULL,
            temperature DOUBLE PRECISION NOT NULL,
            frequency_penalty DOUBLE PRECISION NOT NULL,
            presence_penalty DOUBLE PRECISION NOT NULL,
            repeat_penalty DOUBLE PRECISION NOT NULL,
            min_p DOUBLE PRECISION NOT NULL,
            seed BIGINT
        );

        CREATE TABLE thought_events (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            run_id INT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
            datetime TIMESTAMPTZ NOT NULL,
            thought TEXT NOT NULL,
            target_x INT,
            target_y INT
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
        """
        self.execute(create_sql)

    def _insert_thought(self, event: ThoughtEvent):
        insert_sql ="""
            INSERT INTO thought_events (run_id, datetime, thought, target_x, target_y)
            VALUES (%(run_id)s, %(datetime)s, %(thought)s, %(target_x)s, %(target_y)s)
            """
        
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "thought": event.thought,
            "target_x": event.target[0] if event.target else None,
            "target_y": event.target[1] if event.target else None,
        }
        self.execute(insert_sql, params)

    def _insert_memory_clear(self, event: MemoryClearEvent):
        insert_sql = """
            INSERT INTO memory_clear_events (run_id, datetime, reason)
            VALUES (%(run_id)s, %(datetime)s, %(reason)s)
            """
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "reason": event.reason,
        }
        self.execute(insert_sql, params)

    def _insert_malformed_json(self, event: MalformedJSONEvent):
        insert_sql ="""
            INSERT INTO malformed_json_events (run_id, datetime, content)
            VALUES (%(run_id)s, %(datetime)s, %(content)s)
            """
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "content": event.content,
        }
        self.execute(insert_sql, params)

    def _insert_begin_simulation(self, event: BeginSimulationEvent):
        insert_sql = """
            INSERT INTO runs
            (created_at, context_size, temperature, frequency_penalty,
            presence_penalty, repeat_penalty, min_p, seed)

            VALUES (%(created_at)s, %(context_size)s, %(temperature)s,
            %(frequency_penalty)s, %(presence_penalty)s, %(repeat_penalty)s,
            %(min_p)s, %(seed)s)
            """

        params = {
            "created_at": event.datetime,
            "context_size": event.params.context_size,
            "temperature": event.params.temperature,
            "frequency_penalty": event.params.frequency_penalty,
            "presence_penalty": event.params.presence_penalty,
            "repeat_penalty": event.params.repeat_penalty,
            "min_p": event.params.min_p,
            "seed": event.params.seed,
        }
        self.execute(insert_sql, params)

    def insert(
            self,
            event: ThoughtEvent | MemoryClearEvent | MalformedJSONEvent | BeginSimulationEvent,
        ):
        if isinstance(event, ThoughtEvent):
            self._insert_thought(event)
        elif isinstance(event, MemoryClearEvent):
            self._insert_memory_clear(event)
        elif isinstance(event, MalformedJSONEvent):
            self._insert_malformed_json(event)
        elif isinstance(event, BeginSimulationEvent):
            self._insert_begin_simulation(event)
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
            
    def log(self, event: EventBase):
        if EVENT_LOGGING:
            event.run_id = self.run_id
            self.insert(event) # type: ignore[reportArgumentType]


if __name__ == "__main__":
    with PostGresSQLRecorder(run_id=-1) as rec:
        rec.create_db()