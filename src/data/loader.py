from data.data_formatters import GenericDataFormatter
from data.rossmann.preprocessing import RossmannFormatter
from data.synthetic.preprocessing import SyntheticFormatter
from data.walmart.preprocessing import WalmartFormatter


def data_loader(dataset_name: str) -> GenericDataFormatter:
    if dataset_name == 'rossmann':
        return RossmannFormatter
    elif dataset_name == 'walmart':
        return WalmartFormatter
    elif dataset_name == 'synthetic':
        return SyntheticFormatter
