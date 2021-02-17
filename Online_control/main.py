# Main module
# This module is responsible for initialize and monitor the queues and threads

# real time_domain computer modules
import Aquisicao
import Predicao
import Acionamento
from threading import Thread
from queue import Queue

# create and join the threads
def main():
    # Create the queues to communicate data between modules
    input_analysis_queue = Queue()
    analysis_control_queue = Queue()

    # Create the threads
    th_input = Thread(target=Aquisicao.run, args=(input_analysis_queue,))
    th_analysis = Thread(target=Predicao.run, args=(input_analysis_queue, analysis_control_queue,))
    th_aciona = Thread(target=Acionamento.run, args=(analysis_control_queue,))

    # Start the threads
    th_input.start()
    th_analysis.start()
    th_aciona.start()

if __name__ == "__main__":
    main()
