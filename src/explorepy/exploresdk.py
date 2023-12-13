# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info


# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _exploresdk
else:
    import _exploresdk

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


if _swig_python_version_info[0:2] >= (3, 3):
    import collections.abc
else:
    import collections

class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _exploresdk.delete_SwigPyIterator

    def value(self):
        return _exploresdk.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _exploresdk.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _exploresdk.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _exploresdk.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _exploresdk.SwigPyIterator_equal(self, x)

    def copy(self):
        return _exploresdk.SwigPyIterator_copy(self)

    def next(self):
        return _exploresdk.SwigPyIterator_next(self)

    def __next__(self):
        return _exploresdk.SwigPyIterator___next__(self)

    def previous(self):
        return _exploresdk.SwigPyIterator_previous(self)

    def advance(self, n):
        return _exploresdk.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _exploresdk.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _exploresdk.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _exploresdk.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _exploresdk.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _exploresdk.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _exploresdk.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _exploresdk:
_exploresdk.SwigPyIterator_swigregister(SwigPyIterator)
class BTSerialPortBinding(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_destroy__ = _exploresdk.delete_BTSerialPortBinding

    @staticmethod
    def Create(address, channelID):
        return _exploresdk.BTSerialPortBinding_Create(address, channelID)

    def Connect(self):
        return _exploresdk.BTSerialPortBinding_Connect(self)

    def Close(self):
        return _exploresdk.BTSerialPortBinding_Close(self)

    def Read(self, bt_buffer):
        return _exploresdk.BTSerialPortBinding_Read(self, bt_buffer)

    def Write(self, write_buffer):
        return _exploresdk.BTSerialPortBinding_Write(self, write_buffer)

    def IsDataAvailable(self):
        return _exploresdk.BTSerialPortBinding_IsDataAvailable(self)

# Register BTSerialPortBinding in _exploresdk:
_exploresdk.BTSerialPortBinding_swigregister(BTSerialPortBinding)
class vectordevice(collections.abc.MutableSequence if _swig_python_version_info >= (3, 3) else collections.MutableSequence):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _exploresdk.vectordevice_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _exploresdk.vectordevice___nonzero__(self)

    def __bool__(self):
        return _exploresdk.vectordevice___bool__(self)

    def __len__(self):
        return _exploresdk.vectordevice___len__(self)

    def __getslice__(self, i, j):
        return _exploresdk.vectordevice___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _exploresdk.vectordevice___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _exploresdk.vectordevice___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _exploresdk.vectordevice___delitem__(self, *args)

    def __getitem__(self, *args):
        return _exploresdk.vectordevice___getitem__(self, *args)

    def __setitem__(self, *args):
        return _exploresdk.vectordevice___setitem__(self, *args)

    def pop(self):
        return _exploresdk.vectordevice_pop(self)

    def append(self, x):
        return _exploresdk.vectordevice_append(self, x)

    def empty(self):
        return _exploresdk.vectordevice_empty(self)

    def size(self):
        return _exploresdk.vectordevice_size(self)

    def swap(self, v):
        return _exploresdk.vectordevice_swap(self, v)

    def begin(self):
        return _exploresdk.vectordevice_begin(self)

    def end(self):
        return _exploresdk.vectordevice_end(self)

    def rbegin(self):
        return _exploresdk.vectordevice_rbegin(self)

    def rend(self):
        return _exploresdk.vectordevice_rend(self)

    def clear(self):
        return _exploresdk.vectordevice_clear(self)

    def get_allocator(self):
        return _exploresdk.vectordevice_get_allocator(self)

    def pop_back(self):
        return _exploresdk.vectordevice_pop_back(self)

    def erase(self, *args):
        return _exploresdk.vectordevice_erase(self, *args)

    def __init__(self, *args):
        _exploresdk.vectordevice_swiginit(self, _exploresdk.new_vectordevice(*args))

    def push_back(self, x):
        return _exploresdk.vectordevice_push_back(self, x)

    def front(self):
        return _exploresdk.vectordevice_front(self)

    def back(self):
        return _exploresdk.vectordevice_back(self)

    def assign(self, n, x):
        return _exploresdk.vectordevice_assign(self, n, x)

    def resize(self, *args):
        return _exploresdk.vectordevice_resize(self, *args)

    def insert(self, *args):
        return _exploresdk.vectordevice_insert(self, *args)

    def reserve(self, n):
        return _exploresdk.vectordevice_reserve(self, n)

    def capacity(self):
        return _exploresdk.vectordevice_capacity(self)
    __swig_destroy__ = _exploresdk.delete_vectordevice

# Register vectordevice in _exploresdk:
_exploresdk.vectordevice_swigregister(vectordevice)
class device(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    address = property(_exploresdk.device_address_get, _exploresdk.device_address_set)
    name = property(_exploresdk.device_name_get, _exploresdk.device_name_set)
    lastSeen = property(_exploresdk.device_lastSeen_get, _exploresdk.device_lastSeen_set)
    lastUsed = property(_exploresdk.device_lastUsed_get, _exploresdk.device_lastUsed_set)
    connected = property(_exploresdk.device_connected_get, _exploresdk.device_connected_set)
    remembered = property(_exploresdk.device_remembered_get, _exploresdk.device_remembered_set)
    authenticated = property(_exploresdk.device_authenticated_get, _exploresdk.device_authenticated_set)

    def __init__(self):
        _exploresdk.device_swiginit(self, _exploresdk.new_device())
    __swig_destroy__ = _exploresdk.delete_device

# Register device in _exploresdk:
_exploresdk.device_swigregister(device)
class ExploreSDK(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_destroy__ = _exploresdk.delete_ExploreSDK

    @staticmethod
    def Create():
        return _exploresdk.ExploreSDK_Create()

    def PerformDeviceSearch(self, length=8):
        return _exploresdk.ExploreSDK_PerformDeviceSearch(self, length)

    def SdpSearch(self, address):
        return _exploresdk.ExploreSDK_SdpSearch(self, address)

# Register ExploreSDK in _exploresdk:
_exploresdk.ExploreSDK_swigregister(ExploreSDK)

