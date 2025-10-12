import moabb.datasets as ds
# or
from moabb.datasets import Dreyer2023A

dreyer2023 = Dreyer2023A()
dreyer2023.subject_list = [1, 5, 7, 35]
print(dreyer2023.get_data())  