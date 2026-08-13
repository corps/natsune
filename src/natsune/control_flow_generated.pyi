from dataclasses import dataclass as _dataclass
from natsune.registers import ToInterfaceRegister as _ToInterfaceRegister
from natsune.registers import FromInterfaceRegister as _FromInterfaceRegister


@_dataclass(slots=True)
class FlowInputInto():
    variables: _ToInterfaceRegister
    value: _ToInterfaceRegister


@_dataclass(slots=True)
class FlowInputFrom():
    variables: _FromInterfaceRegister
    value: _FromInterfaceRegister


@_dataclass(slots=True)
class FlowControlInto():
    return_value: _FromInterfaceRegister
    continue_variables: _FromInterfaceRegister
    break_variables: _FromInterfaceRegister
    finish_variables: _FromInterfaceRegister


@_dataclass(slots=True)
class FlowControlFrom():
    return_value: _ToInterfaceRegister
    continue_variables: _ToInterfaceRegister
    break_variables: _ToInterfaceRegister
    finish_variables: _ToInterfaceRegister


@_dataclass(slots=True)
class IfThenElseOutputInto():
    context: _ToInterfaceRegister
    result: _FromInterfaceRegister


@_dataclass(slots=True)
class IfThenElseOutputFrom():
    context: _FromInterfaceRegister
    result: _ToInterfaceRegister


@_dataclass(slots=True)
class MergeOutputInto():
    second_value: _ToInterfaceRegister
    result: _FromInterfaceRegister


@_dataclass(slots=True)
class MergeOutputFrom():
    second_value: _FromInterfaceRegister
    result: _ToInterfaceRegister
