from .database import db_session
from .models import SinglePhase
from .regression_model import *
from .dl_model import *
from celery import Celery
from sqlalchemy import inspect
from IPython.utils.capture import capture_output

celery = Celery("tasks", backend='rpc://',
                broker='amqp://guest:guest@35.196.180.90:5672', queue="analysis")


@celery.task(name="analysis.tasks.analyze")
def analyze(calculation_id):

    sorted_machine_digital_state_data, sorted_machine_analog_phase_data = collect_phase_data(calculation_id)

    with capture_output() as captured:

        train_regression_model(sorted_machine_digital_state_data, sorted_machine_analog_phase_data)
        train_deep_learning_model(package_for_dl(sorted_machine_digital_state_data, sorted_machine_analog_phase_data))

    result = captured.stdout

    return result


def package_for_dl(sorted_machine_digital_state_data, sorted_machine_analog_phase_data):
    return {'train':
                {
                    'phases': sorted_machine_analog_phase_data,
                    'states': sorted_machine_digital_state_data
                },
            'test': {
                    'phases': sorted_machine_analog_phase_data,
                    'states': sorted_machine_digital_state_data
                }
    }


def collect_phase_data(calculation_id):
    phases = SinglePhase.query.filter_by(calculation_id=calculation_id).all()
    data = {}

    initial_phase = None

    for phase in phases:
        phase = object_as_dict(phase)
        data[phase['previous_phase_id']] = phase
        if phase['previous_phase_id'] == -1:
            initial_phase = phase

    sorted_machine_digital_state_data = []
    sorted_machine_analog_phase_data = []
    current_node = initial_phase

    while len(sorted_machine_digital_state_data) < len(data):
        sorted_machine_digital_state_data.append(int(current_node['states']))  # discarding flags
        sorted_machine_analog_phase_data.append(int(current_node['phase']))
        next_node = data[current_node['id']]
        current_node = next_node

    return sorted_machine_digital_state_data, sorted_machine_analog_phase_data


def add_refresh(obj):
    db_session.add(obj)
    db_session.flush()
    db_session.refresh(obj)


def object_as_dict(obj):
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}
