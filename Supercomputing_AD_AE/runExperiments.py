import os
import re
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

BINDIR= "/gpfs/alpine/csc333/proj-shared/"
bin_commands = {
    "alphaOMP": BINDIR + "ArborX_dendrogram_crusher_openmp --algorithm hdbscan --dendrogram alpha --verbose --binary --filename ",
    "unionFindOMP": BINDIR + "ArborX_dendrogram_crusher_openmp --algorithm hdbscan --dendrogram union-find --verbose --binary --filename ",
    "alphaHIP": BINDIR + "ArborX_dendrogram_crusher_hip --algorithm hdbscan --dendrogram alpha --verbose --binary --filename "
}

DataSets = [ "2D_VisualSim_10M.arborx",
                "3D_VisualSim_10M.arborx",
                "2D_VisualVar_10M.arborx",
                "3D_VisualVar_10M.arborx",
                "5D_VisualSim_10M.arborx",
                "5D_VisualVar_10M.arborx",
                # "3D_GeoLife_24M.arborx",
                "ngsim.arborx",
                "3DRoadNetwork.arborx",
                "hacc_37M.arborx",
                "PortoTaxi.arborx",
                "ngsim_location3.arborx",
                "normal10M2.arborx",
                "uniform100M2.arborx",
                "normal100M2.arborx",
                "normal10M3.arborx",
                "uniform100M3.arborx",
                "normal100M3.arborx"]

DSpath = [ BINDIR +dataFile for dataFile in DataSets]

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiment'
    experiment_id = Column(Integer, primary_key=True)
    dataset = Column(String)
    bin_type = Column(String)
    num_trials = Column(Integer)
    status = Column(String)
    observations = relationship("Observation", back_populates="experiment")

class Observation(Base):
    __tablename__ = 'observation'
    observation_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiment.experiment_id'))
    trial_id = Column(Integer)
    mst_time = Column(Float)
    dendrogram_time = Column(Float)
    edge_sort_time = Column(Float, nullable=True)
    alpha_edges_time = Column(Float, nullable=True)
    alpha_vertices_time = Column(Float, nullable=True)
    alpha_matrix_time = Column(Float, nullable=True)
    sided_parents_time = Column(Float, nullable=True)
    compression_time = Column(Float, nullable=True)
    parents_time = Column(Float, nullable=True)
    total_time = Column(Float)
    error_message = Column(String, nullable=True)
    log_file = Column(String)
    experiment = relationship("Experiment", back_populates="observations")

def create_database(db_name='sqlite:///experiments.db'):
    engine = create_engine(db_name)
    Base.metadata.create_all(engine)
    return engine

def create_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

def run_experiment(dataset, bin_type, command, trial_id, log_folder):
    log_file = f"{log_folder}/{dataset}_{bin_type}_{trial_id}.log"
    with open(log_file, "w") as log_f:
        try:
            # ... (same as before)
            print(f"Running: {command} {BINDIR}{dataset}")
            result = subprocess.run(f"{command} {BINDIR}{dataset}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            log_f.write(result.stdout)
            log_f.write(result.stderr)
            
            regex_values = {
                "mst_time": r"mst\s+:\s+(\d+\.\d+)",
                "dendrogram_time": r"dendrogram\s+:\s+(\d+\.\d+)",
                "edge_sort_time": r"edge sort\s+:\s+(\d+\.\d+)",
                "alpha_edges_time": r"alpha edges\s+:\s+(\d+\.\d+)",
                "alpha_vertices_time": r"alpha vertices\s+:\s+(\d+\.\d+)",
                "alpha_matrix_time": r"alpha matrix\s+:\s+(\d+\.\d+)",
                "sided_parents_time": r"sided parents\s+:\s+(\d+\.\d+)",
                "compression_time": r"compression\s+:\s+(\d+\.\d+)",
                "parents_time": r"parents\s+:\s+(\d+\.\d+)",
                "total_time": r"total time\s+:\s+(\d+\.\d+)"
            }
            extracted_values = {}
            for key, pattern in regex_values.items():
                match = re.search(pattern, result.stdout)
                if match:
                    extracted_values[key] = float(match.group(1))
            return extracted_values, None, log_file
        except subprocess.CalledProcessError as e:
            # ... (same as before)
            return None, str(e), log_file

def store_experiment_results(session, dataset, bin_type, trial_id, num_trials, log_folder, command):
 
    
    experiment = (
        session.query(Experiment)
        .filter_by(dataset=dataset, bin_type=bin_type)
        .one_or_none()
    )

    if not experiment:
        experiment = Experiment(dataset=dataset, bin_type=bin_type, num_trials=num_trials, status="in_progress")
        session.add(experiment)
        session.commit()

    
    existing_observation = (
        session.query(Observation)
        .filter_by(experiment_id=experiment.experiment_id, trial_id=trial_id)
        .one_or_none()
    )

    if existing_observation:
        print(f"Skipping dataset: {dataset}, bin_type: {bin_type}, trial_id: {trial_id} (already exists)")
        return 


    output_values, error_message, log_file = run_experiment(dataset, bin_type, command, trial_id, log_folder)
    observation = Observation(
        experiment_id=experiment.experiment_id,
        trial_id=trial_id,
        mst_time=output_values.get("mst_time") if output_values else None,
        # ... (same as before)
        error_message=error_message,
        log_file=log_file
    )
    session.add(observation)
    session.commit()

    experiment.status = f"completed {trial_id + 1}/{num_trials}"
    session.commit()


def run_and_store_experiments(DataSets, bin_commands, num_trials=3, db_name='sqlite:///experiments.db', log_folder="log"):
    engine = create_database(db_name)
    session = create_session(engine)
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    for trial_id in range(num_trials):
        for dataset in DataSets:
            for bin_type, command in bin_commands.items():
                store_experiment_results(session, dataset, bin_type, trial_id, num_trials, log_folder, command)

    session.close()


if __name__ == "__main__":
    # Same DataSets and bin_commands as before
    run_and_store_experiments(DataSets, bin_commands)