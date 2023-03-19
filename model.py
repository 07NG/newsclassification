import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize 
from sklearn import preprocessing 

import re
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('stopwords') #download stopwords from nltk corpus
nltk.download('punkt') #download punkt from nltk corpus

data = pd.read_csv("data.csv" ) #read data.csv file into Pandas dataframe 'data'

#data cleaning 
def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip() #Convert the input text to lowercase, remove newlines, carriage returns 
    text = re.sub(' +', ' ', text)# remove extra white spaces
    text = re.sub(r'[^\w\s]','',text)# remove all non-word characters #\w->special char #\s->spcae or tab
    
    
    stop_words = set(stopwords.words('english')) #create sets of stopwords from nltk corpus into 'stop_words'
    word_tokens = word_tokenize(text) #Tokenize the cleaned text into individual words
    filtered_sentence = [w for w in word_tokens if not w in stop_words] #Remove all stop words from the tokenized text
    # filtered_sentence = [] 
    # for w in word_tokens: 
    #     if w not in stop_words: 
    #         filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence) #Join the remaining words back into a single string
    return text #Return the processed text

#assining cleaned data[text] into data[textparsed]
data['Text_parsed'] = data['Text'].apply(process_text) #cleaned data[text] into data[textparsed]



label_encoder = preprocessing.LabelEncoder() #create instance of labelencoder
data['Category_target']= label_encoder.fit_transform(data['Category']) #transforms category column into unique numeric values


X_train, X_test, y_train, y_test = train_test_split(data['Text_parsed'], # split testing set to be 20% for category and parsed data
                                                    data['Category_target'], 
                                                    test_size=0.2, 
                                                    random_state=8)

ngram_range = (1,2) #sets ranger of ngram[sequence of words] to unigram-"i" and bigram-"i have" 
min_df = 10 #sets min document frequency to 10
max_df = 1. #sets max document frequency to 1
max_features = 300 #top 300 features with highest idf[rare or unique] scores will be included

tfidf = TfidfVectorizer(encoding='utf-8',
                        decode_error='ignore',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True) #create instace of tdifevectorizer for feature selection
                        
features_train = tfidf.fit_transform(X_train).toarray() #transform x-train to feature matrix

#FEATURE EXTRACTOR DUMPED
pickle.dump(tfidf,open('feature.pkl','wb')) #saves trained tdif into binary file i.e,feature.pkl

labels_train = y_train #assigns labels-train with y-train containing target variables i.e,category for training data

features_test = tfidf.transform(X_test).toarray() #feature matrix into numpy array
labels_test = y_test #assigns lables-test with y-test containing target variable i.e,category  for testing data


LR = LogisticRegression(C=1) #intialize LRC with regularization strength of c=1
LR.fit(features_train, labels_train) #trains LRC by fitting target variables and feature matrix

LR = DecisionTreeClassifier() #initalizes decisiontreeclassifier
LR.fit(features_train, labels_train) #trains dtc by fitting target variables and feature matrix

#PREDICTIONS

#THIS IS THE INPUT TO THE MODEL
#input = pd.read_csv("input.csv",encoding_errors='replace')
#processed input
#processed_input = input.apply(process_text)

#EXTRACTING THE FEATURES FROM THE INPUT DATA
#input_features = tfidf.transform(processed_input).toarray()

#prediction = LR.predict(input_features)
#print(prediction)

pickle.dump(LR,open('model.pkl','wb')) #saves trained LRC in binary file:model.pkl

# model = pickle.load(open('model.pkl','rb'))

class TfidfVectorizerClass(CountVectorizer): #tdifvectorizer class inheriting countvectorizer

    def __init__( #initalize parameters into constructor
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        super().__init__( # calls constructor of base class and initalize with parameters
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._tfidf = TfidfTransformer( #compute scores and all argument are property defined below and private instance variable self._tfidf
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property #property declaration
    def norm(self): #creating norm property
        """Norm of each row output, can be either "l1" or "l2"."None"""
        return self._tfidf.norm

    @norm.setter #setter method for norm
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        """Whether or not IDF re-weighting is used."""
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        """Whether or not IDF weights are smoothed."""
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        """Whether or not sublinear TF scaling is applied."""
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.

        Returns
        -------
        ndarray of shape (n_features,)
        """
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal to vocabulary size = %d"
                    % (len(value), len(self.vocabulary))
                )
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is not needed to compute tfidf.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self._check_params()
        self._warn_for_unused_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        y : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}

class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty:

        - `'none'`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.

        .. warning::
           The choice of the algorithm depends on the penalty chosen:
           Supported penalties by solver:

           - 'newton-cg'   -   ['l2', 'none']
           - 'lbfgs'       -   ['l2', 'none']
           - 'liblinear'   -   ['l1', 'l2']
           - 'sag'         -   ['l2', 'none']
           - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on
           features with approximately the same scale. You can
           preprocess the data with a scaler from :mod:`sklearn.preprocessing`.

        .. seealso::
           Refer to the User Guide for more information regarding
           :class:`LogisticRegression` and more specifically the
           `Table <https://scikit-learn.org/dev/modules/linear_model.html#logistic-regression>`_
           summarazing solver/penalty supports.
           <!--
           # noqa: E501
           -->

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
        SAGA: A Fast Incremental Gradient Method With Support
        for Non-Strongly Convex Composite Objectives
        https://arxiv.org/abs/1407.0202

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    """

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        The SAGA solver supports both float64 and float32 bit arrays.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        if self.penalty == "elasticnet":
            if (
                not isinstance(self.l1_ratio, numbers.Number)
                or self.l1_ratio < 0
                or self.l1_ratio > 1
            ):
                raise ValueError(
                    "l1_ratio must be between 0 and 1; got (l1_ratio=%r)"
                    % self.l1_ratio
                )
        elif self.l1_ratio is not None:
            warnings.warn(
                "l1_ratio parameter is only used when penalty is "
                "'elasticnet'. Got "
                "(penalty={})".format(self.penalty)
            )
        if self.penalty == "none":
            if self.C != 1.0:  # default values
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        if solver == "lbfgs":
            _dtype = np.float64
        else:
            _dtype = [np.float64, np.float32]

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        if solver == "liblinear":
            if effective_n_jobs(self.n_jobs) != 1:
                warnings.warn(
                    "'n_jobs' > 1 does not have any effect when"
                    " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                    " = {}.".format(effective_n_jobs(self.n_jobs))
                )
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X,
                y,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                self.class_weight,
                self.penalty,
                self.dual,
                self.verbose,
                self.max_iter,
                self.tol,
                self.random_state,
                sample_weight=sample_weight,
            )
            self.n_iter_ = np.array([n_iter_])
            return self

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
            )

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == "multinomial":
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"
        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
                y,
                pos_class=class_,
                Cs=[C_],
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                verbose=self.verbose,
                solver=solver,
                multi_class=multi_class,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                check_input=False,
                random_state=self.random_state,
                coef=warm_start_coef_,
                penalty=penalty,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
            )
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
        )

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        n_features = X.shape[1]
        if multi_class == "multinomial":
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(
                n_classes, n_features + int(self.fit_intercept)
            )

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        ovr = self.multi_class in ["ovr", "warn"] or (
            self.multi_class == "auto"
            and (self.classes_.size <= 2 or self.solver == "liblinear")
        )
        if ovr:
            return super()._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


class LogisticRegressionCVClass(LogisticRegression, LinearClassifierMixin, BaseEstimator):

    def __init__(
        self,
        *,
        Cs=10,
        fit_intercept=True,
        cv=None,
        dual=False,
        penalty="l2",
        scoring=None,
        solver="lbfgs",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        n_jobs=None,
        verbose=0,
        refit=True,
        intercept_scaling=1.0,
        multi_class="auto",
        random_state=None,
        l1_ratios=None,
    ):
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
        self.l1_ratios = l1_ratios

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Fitted LogisticRegressionCV estimator.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )
        if self.penalty == "elasticnet":
            if (
                self.l1_ratios is None
                or len(self.l1_ratios) == 0
                or any(
                    (
                        not isinstance(l1_ratio, numbers.Number)
                        or l1_ratio < 0
                        or l1_ratio > 1
                    )
                    for l1_ratio in self.l1_ratios
                )
            ):
                raise ValueError(
                    "l1_ratios must be a list of numbers between "
                    "0 and 1; got (l1_ratios=%r)"
                    % self.l1_ratios
                )
            l1_ratios_ = self.l1_ratios
        else:
            if self.l1_ratios is not None:
                warnings.warn(
                    "l1_ratios parameter is only used when penalty "
                    "is 'elasticnet'. Got (penalty={})".format(self.penalty)
                )

            l1_ratios_ = [None]

        if self.penalty == "none":
            raise ValueError(
                "penalty='none' is not useful and not supported by "
                "LogisticRegressionCV."
            )

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)

        class_weight = self.class_weight

        # Encode for string labels
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        if isinstance(class_weight, dict):
            class_weight = {
                label_encoder.transform([cls])[0]: v for cls, v in class_weight.items()
            }

        # The original class labels
        classes = self.classes_ = label_encoder.classes_
        encoded_labels = label_encoder.transform(label_encoder.classes_)

        multi_class = _check_multi_class(self.multi_class, solver, len(classes))

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        # init cross-validation generator
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))

        # Use the label encoded classes
        n_classes = len(encoded_labels)

        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes[0]
            )

        if n_classes == 2:
            # OvR in case of binary problems is as good as fitting
            # the higher label
            n_classes = 1
            encoded_labels = encoded_labels[1:]
            classes = classes[1:]

        # We need this hack to iterate only once over labels, in the case of
        # multi_class = multinomial, without changing the value of the labels.
        if multi_class == "multinomial":
            iter_encoded_labels = iter_classes = [None]
        else:
            iter_encoded_labels = encoded_labels
            iter_classes = classes

        # compute the class weights for the entire dataset y
        if class_weight == "balanced":
            class_weight = compute_class_weight(
                class_weight, classes=np.arange(len(self.classes_)), y=y
            )
            class_weight = dict(enumerate(class_weight))

        path_func = delayed(_log_reg_scoring_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"

        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
                y,
                train,
                test,
                pos_class=label,
                Cs=self.Cs,
                fit_intercept=self.fit_intercept,
                penalty=self.penalty,
                dual=self.dual,
                solver=solver,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
                class_weight=class_weight,
                scoring=self.scoring,
                multi_class=multi_class,
                intercept_scaling=self.intercept_scaling,
                random_state=self.random_state,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
                l1_ratio=l1_ratio,
            )
            for label in iter_encoded_labels
            for train, test in folds
            for l1_ratio in l1_ratios_
        )

        # _log_reg_scoring_path will output different shapes depending on the
        # multi_class param, so we need to reshape the outputs accordingly.
        # Cs is of shape (n_classes . n_folds . n_l1_ratios, n_Cs) and all the
        # rows are equal, so we just take the first one.
        # After reshaping,
        # - scores is of shape (n_classes, n_folds, n_Cs . n_l1_ratios)
        # - coefs_paths is of shape
        #  (n_classes, n_folds, n_Cs . n_l1_ratios, n_features)
        # - n_iter is of shape
        #  (n_classes, n_folds, n_Cs . n_l1_ratios) or
        #  (1, n_folds, n_Cs . n_l1_ratios)
        coefs_paths, Cs, scores, n_iter_ = zip(*fold_coefs_)
        self.Cs_ = Cs[0]
        if multi_class == "multinomial":
            coefs_paths = np.reshape(
                coefs_paths,
                (len(folds), len(l1_ratios_) * len(self.Cs_), n_classes, -1),
            )
            # equiv to coefs_paths = np.moveaxis(coefs_paths, (0, 1, 2, 3),
            #                                                 (1, 2, 0, 3))
            coefs_paths = np.swapaxes(coefs_paths, 0, 1)
            coefs_paths = np.swapaxes(coefs_paths, 0, 2)
            self.n_iter_ = np.reshape(
                n_iter_, (1, len(folds), len(self.Cs_) * len(l1_ratios_))
            )
            # repeat same scores across all classes
            scores = np.tile(scores, (n_classes, 1, 1))
        else:
            coefs_paths = np.reshape(
                coefs_paths,
                (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_), -1),
            )
            self.n_iter_ = np.reshape(
                n_iter_, (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_))
            )
        scores = np.reshape(scores, (n_classes, len(folds), -1))
        self.scores_ = dict(zip(classes, scores))
        self.coefs_paths_ = dict(zip(classes, coefs_paths))

        self.C_ = list()
        self.l1_ratio_ = list()
        self.coef_ = np.empty((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        for index, (cls, encoded_label) in enumerate(
            zip(iter_classes, iter_encoded_labels)
        ):

            if multi_class == "ovr":
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]
            else:
                # For multinomial, all scores are the same across classes
                scores = scores[0]
                # coefs_paths will keep its original shape because
                # logistic_regression_path expects it this way

            if self.refit:
                # best_index is between 0 and (n_Cs . n_l1_ratios - 1)
                # for example, with n_cs=2 and n_l1_ratios=3
                # the layout of scores is
                # [c1, c2, c1, c2, c1, c2]
                #   l1_1 ,  l1_2 ,  l1_3
                best_index = scores.sum(axis=0).argmax()

                best_index_C = best_index % len(self.Cs_)
                C_ = self.Cs_[best_index_C]
                self.C_.append(C_)

                best_index_l1 = best_index // len(self.Cs_)
                l1_ratio_ = l1_ratios_[best_index_l1]
                self.l1_ratio_.append(l1_ratio_)

                if multi_class == "multinomial":
                    coef_init = np.mean(coefs_paths[:, :, best_index, :], axis=1)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)

                # Note that y is label encoded and hence pos_class must be
                # the encoded label / None (for 'multinomial')
                w, _, _ = _logistic_regression_path(
                    X,
                    y,
                    pos_class=encoded_label,
                    Cs=[C_],
                    solver=solver,
                    fit_intercept=self.fit_intercept,
                    coef=coef_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    penalty=self.penalty,
                    class_weight=class_weight,
                    multi_class=multi_class,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    check_input=False,
                    max_squared_sum=max_squared_sum,
                    sample_weight=sample_weight,
                    l1_ratio=l1_ratio_,
                )
                w = w[0]

            else:
                # Take the best scores across every fold and the average of
                # all coefficients corresponding to the best scores.
                best_indices = np.argmax(scores, axis=1)
                if multi_class == "ovr":
                    w = np.mean(
                        [coefs_paths[i, best_indices[i], :] for i in range(len(folds))],
                        axis=0,
                    )
                else:
                    w = np.mean(
                        [
                            coefs_paths[:, i, best_indices[i], :]
                            for i in range(len(folds))
                        ],
                        axis=0,
                    )

                best_indices_C = best_indices % len(self.Cs_)
                self.C_.append(np.mean(self.Cs_[best_indices_C]))

                if self.penalty == "elasticnet":
                    best_indices_l1 = best_indices // len(self.Cs_)
                    self.l1_ratio_.append(np.mean(l1_ratios_[best_indices_l1]))
                else:
                    self.l1_ratio_.append(None)

            if multi_class == "multinomial":
                self.C_ = np.tile(self.C_, n_classes)
                self.l1_ratio_ = np.tile(self.l1_ratio_, n_classes)
                self.coef_ = w[:, : X.shape[1]]
                if self.fit_intercept:
                    self.intercept_ = w[:, -1]
            else:
                self.coef_[index] = w[: X.shape[1]]
                if self.fit_intercept:
                    self.intercept_[index] = w[-1]

        self.C_ = np.asarray(self.C_)
        self.l1_ratio_ = np.asarray(self.l1_ratio_)
        self.l1_ratios_ = np.asarray(l1_ratios_)
        # if elasticnet was used, add the l1_ratios dimension to some
        # attributes
        if self.l1_ratios is not None:
            # with n_cs=2 and n_l1_ratios=3
            # the layout of scores is
            # [c1, c2, c1, c2, c1, c2]
            #   l1_1 ,  l1_2 ,  l1_3
            # To get a 2d array with the following layout
            #      l1_1, l1_2, l1_3
            # c1 [[ .  ,  .  ,  .  ],
            # c2  [ .  ,  .  ,  .  ]]
            # We need to first reshape and then transpose.
            # The same goes for the other arrays
            for cls, coefs_path in self.coefs_paths_.items():
                self.coefs_paths_[cls] = coefs_path.reshape(
                    (len(folds), self.l1_ratios_.size, self.Cs_.size, -1)
                )
                self.coefs_paths_[cls] = np.transpose(
                    self.coefs_paths_[cls], (0, 2, 1, 3)
                )
            for cls, score in self.scores_.items():
                self.scores_[cls] = score.reshape(
                    (len(folds), self.l1_ratios_.size, self.Cs_.size)
                )
                self.scores_[cls] = np.transpose(self.scores_[cls], (0, 2, 1))

            self.n_iter_ = self.n_iter_.reshape(
                (-1, len(folds), self.l1_ratios_.size, self.Cs_.size)
            )
            self.n_iter_ = np.transpose(self.n_iter_, (0, 1, 3, 2))

        return self

    def score(self, X, y, sample_weight=None):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        scoring = self.scoring or "accuracy"
        scoring = get_scorer(scoring)

        return scoring(self, X, y, sample_weight=sample_weight)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
