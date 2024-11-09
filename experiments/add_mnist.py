# Experiment: accuracy curve for Add MNIST.
import pasp, numpy as np

def init():
  "Initializes test labels, and accuracy lists."
  import torchvision
  test_labels = torchvision.datasets.MNIST(root="/tmp", train=False, \
                                                download=True).targets.data.numpy()
  test_both_labels = test_labels[:(h := len(test_labels)//2)] + \
                          test_labels[h:]
  accuracy = []
  accuracy_program = []
  return test_labels, test_both_labels, accuracy, accuracy_program

if __name__ == "__main__":
  L, L_b, A, A_p = init()
  P = pasp.parse("examples/add_mnist.plp")

  def step(self):
    "Step callback function for each iteration of training."
    Y = np.argmax(self.pr(), axis=1)
    A.append(np.sum(Y == L)/len(Y))
    Y_b = Y[:(h := len(Y)//2)] + Y[h:]
    A_p.append(np.sum(Y_b == L_b)/len(Y_b))

  # Pass step as the step callback function.
  P.NA[0].set_step_callback(step)
  # Run the program to learn parameters.
  P()
  # A and A_p are accuracies of embedded neural network, and program (sum).
  A, A_p = np.array(A), np.array(A_p)
  print(A)
  print(A_p)
