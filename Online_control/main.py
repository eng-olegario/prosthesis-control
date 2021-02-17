# Main module
# This module is responsible for initialize and monitor the queues and threads

# real time_domain computer modules
import Acquisition
import Prediction
import Actuation
from threading import Thread
from queue import Queue

# create and join the threads
def main():
    # Create the queues to communicate data between modules
    acquisition_prediction_queue = Queue()
    prediction_control_queue = Queue()

    # Create the threads
    th_acquisition = Thread(target=Acquisition.run, args=(acquisition_prediction_queue,))
    th_prediction = Thread(target=Prediction.run, args=(acquisition_prediction_queue, prediction_control_queue,))
    th_actuation = Thread(target=Actuation.run, args=(prediction_control_queue,))

    # Start the threads
    th_acquisition.start()
    th_prediction.start()
    th_actuation.start()

if __name__ == "__main__":
    main()
