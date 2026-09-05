from pydantic import BaseModel, Field
from lib.types.config import SimulationConfig, LossFunctionWeights, TunerConfig, ParamsConfig

class BrainReport(BaseModel):
    iterations: int = Field(0, description="The total number of request/responses by the LLM")
    thought_loops: int = Field(0, description="The number of thought loops detected")
    empty_thoughts: int = Field(0, description="The number of times the LLM has had an empty thought")
    out_of_bounds_attempts: int = Field(0, description="The number of times the LLM has attempted to go out of bounds")
    malformed_json: int = Field(0, description="The number of times the LLM returned unparseable JSON")
    non_alphanumeric: int = Field(0, description="The number of times the LLM has had a non-alphanumeric thought")
    actual_runtime: None | float = Field(None, description="The actual runtime of the simulation in seconds")

    @classmethod
    def average(cls, reports: list["BrainReport"]) -> "BrainReport":
        """Calculates the arithmetic mean of a list of BrainReport instances.

        Args:
            reports: A list of BrainReport instances to average.

        Returns:
            A new BrainReport instance containing the averaged metric values.
        """
        if not reports:
            raise ValueError(f"expected values")

        total_reports = len(reports)
        valid_runtimes = [
            report.actual_runtime
            for report in reports
            if report.actual_runtime is not None
        ]

        def avg(item:str):
            return round(sum(getattr(report, item) for report in reports) / total_reports)

        return cls(
            iterations=avg("iterations"),
            thought_loops=avg("thought_loops"),
            empty_thoughts=avg("empty_thoughts"),
            out_of_bounds_attempts=avg("out_of_bounds_attempts"),
            malformed_json=avg("malformed_json"),
            non_alphanumeric=avg("non_alphanumeric"),
            actual_runtime=(sum(valid_runtimes) / len(valid_runtimes) if valid_runtimes else None)
            )
    
class OutputReport(BaseModel):
    config: SimulationConfig
    report: BrainReport

class Trial(BaseModel):
    params: ParamsConfig
    report: BrainReport

class TrialCollection(BaseModel):
    params: ParamsConfig
    seeds: list[int|None]
    reports: list[BrainReport]

class StudyReport(BaseModel):
    comments: str = Field("")
    tuner_config: TunerConfig
    loss_function_weights: LossFunctionWeights
    simulation_config: SimulationConfig
    trials: list[Trial]


class StudyReportCollection(BaseModel):
    comments: str = Field("")
    tuner_config: TunerConfig
    loss_function_weights: LossFunctionWeights
    simulation_config: SimulationConfig
    trials: list[TrialCollection]

    @classmethod
    def collect(cls, data: StudyReport) -> "StudyReportCollection":
        """Converts a StudyReport into a StudyReportCollection by grouping trials with matching parameters.
        """
        grouped_trials: dict[str, TrialCollection] = {}

        for trial in data.trials:
            normalized_params = trial.params.model_copy(update={"seed": None})
            group_key = normalized_params.model_dump_json()

            if group_key not in grouped_trials:
                grouped_trials[group_key] = TrialCollection(
                    params=normalized_params,
                    seeds=[],
                    reports=[],
                )

            grouped_trials[group_key].seeds.append(trial.params.seed)
            grouped_trials[group_key].reports.append(trial.report)

        return cls(
            comments=data.comments,
            tuner_config=data.tuner_config,
            loss_function_weights=data.loss_function_weights,
            simulation_config=data.simulation_config,
            trials=list(grouped_trials.values()),
        )

    def get_average(self) -> StudyReport:
        """Calculates the average BrainReport for each unique set of parameters in the StudyReportCollection."""
        trials: list[Trial] = []
        for collection in self.trials:
            trials.append(
                Trial(
                    params=collection.params,
                    report=BrainReport.average(collection.reports))
            )

        return StudyReport(
            comments=self.comments,
            tuner_config=self.tuner_config,
            loss_function_weights=self.loss_function_weights,
            simulation_config=self.simulation_config,
            trials=trials
        )

    @classmethod
    def average_study(cls, study: StudyReport) -> StudyReport:
        return cls.collect(study).get_average()
