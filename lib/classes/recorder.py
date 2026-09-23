from pathlib import Path
from abc import ABCMeta, abstractmethod
from uuid import uuid4

from py_pglite.sqlalchemy import SQLAlchemyPGliteManager # type: ignore[import-untyped]
from sqlalchemy import Connection, Engine, text

from lib.types.log import EventBase, ThoughtEvent, MemoryClearEvent, MalformedJSONEvent, BeginSimulationEvent

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
        event.run_id = self.run_id
        if EVENT_LOGGING:
            if not self.path.exists():
                self.path.touch()
            
            with open(self.path, "a") as f:
                f.write(event.model_dump_json() + "\n")

class PostGresSQLRecorder(RecorderBase):

    def __init__(self, path: Path, run_id: int|None = None):
        super().__init__(run_id=run_id, path=path)

    def create_db(self, engine: Engine):
        create_sql = text("""
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
            min_p DOUBLE PRECISION NOT NULL DEFAULT,
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
        """)
        with engine.connect() as conn:
            conn.execute(create_sql)

    def _insert_thought(self, event: ThoughtEvent, connection: Connection):
        insert_sql = text(
            """
            INSERT INTO thought_events (run_id, datetime, thought, target_x, target_y)
            VALUES (:run_id, :datetime, :thought, :target_x, :target_y)
            """
        )
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "thought": event.thought,
            "target_x": event.target[0] if event.target else None,
            "target_y": event.target[1] if event.target else None,
        }
        connection.execute(insert_sql, params)

    def _insert_memory_clear(self, event: MemoryClearEvent, connection: Connection):
        insert_sql = text(
            """
            INSERT INTO memory_clear_events (run_id, datetime, reason)
            VALUES (:run_id, :datetime, :reason)
            """
        )
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "reason": event.reason.value,
        }
        connection.execute(insert_sql, params)

    def _insert_malformed_json(self, event: MalformedJSONEvent, connection: Connection):
        insert_sql = text(
            """
            INSERT INTO malformed_json_events (run_id, datetime, content)
            VALUES (:run_id, :datetime, :content)
            """
        )
        params = {
            "run_id": event.run_id,
            "datetime": event.datetime,
            "content": event.content,
        }
        connection.execute(insert_sql, params)
        connection.commit()

    def _insert_begin_simulation(self, event: BeginSimulationEvent, connection: Connection):
        insert_sql = text(
            """
            INSERT INTO runs (created_at, context_size, temperature, frequency_penalty, presence_penalty, repeat_penalty, min_p, seed)
            VALUES (:created_at, :context_size, :temperature, :frequency_penalty, :presence_penalty, :repeat_penalty, :min_p, :seed)
            """
        )
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
        connection.execute(insert_sql, params)

    def insert(
            self,
            event: ThoughtEvent | MemoryClearEvent | MalformedJSONEvent | BeginSimulationEvent,
            engine: Engine
        ):
        with engine.connect() as conn:
            if isinstance(event, ThoughtEvent):
                self._insert_thought(event, conn)
            elif isinstance(event, MemoryClearEvent):
                self._insert_memory_clear(event, conn)
            elif isinstance(event, MalformedJSONEvent):
                self._insert_malformed_json(event, conn)
            elif isinstance(event, BeginSimulationEvent):
                self._insert_begin_simulation(event, conn)
            else:
                raise ValueError(f"Unsupported event type: {type(event)}")
            

class PGLiteSQLRecorder(PostGresSQLRecorder):
    # this is a pretty bizarre stack (py-pglite -> SQLAlchemy -> raw SQL -> pglite -> postgres)
    # because I wanted to practice postgres commands while being able to commit the db to git
    def __init__(self, path: Path, run_id: int|None = None):
        super().__init__(run_id=run_id, path=path)

    def log(self, event: EventBase):
        if EVENT_LOGGING:
            event.run_id = self.run_id
            with SQLAlchemyPGliteManager() as db:
                engine: Engine = db.get_engine()

                if not self.path.exists():
                    self.create_db(engine)

                self.insert(event, engine) # type: ignore[reportArgumentType]
