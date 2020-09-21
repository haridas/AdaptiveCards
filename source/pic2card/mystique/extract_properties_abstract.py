import abc


class AbstractBaseExtractProperties(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_text(self, image=None, coords=None):
        pass

    @abc.abstractmethod
    def get_alignment(self, image=None, xmin=None, xmax=None):
        pass


class AbstractFontSize(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_size(self, image=None, coords=None, img_data=None):
        pass


class AbstractFontWeight(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_weight(self, image=None, coords=None):
        pass


class AbstractChoiceExtraction(AbstractBaseExtractProperties):
    pass


class AbstractTextExtraction(AbstractBaseExtractProperties, AbstractFontSize,
                             AbstractFontWeight):

    @abc.abstractmethod
    def get_colors(self, image=None, coords=None):
        pass


"""
class AbstractActionsetExtraction(AbstractBaseExtractProperties):

    @abc.abstractmethod
    def get_actionset_type(self, image=None, coords=None):
        pass


class AbstractImageExtraction(AbstractBaseExtractProperties):

    @abc.abstractmethod
    def image(self, image=None, coords=None):
        pass
"""
