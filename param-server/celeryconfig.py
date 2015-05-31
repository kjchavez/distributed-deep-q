from datetime import timedelta

CELERYBEAT_SCHEDULE = {
    'evaluate_reward': {
        'task': 'tasks.evaluateReward',
        'schedule': timedelta(seconds=30),
        'args': (),
    },
    'save_reward_to_disk': {
        'task': 'tasks.saveAverageReward',
        'schedule': timedelta(seconds=600),
        'args': (),
    },
}


MODEL_FILE = '../models/deepq/deepq16.caffemodel'
ARCHITECTURE_FILE = '../models/deepq/train_val.prototxt'