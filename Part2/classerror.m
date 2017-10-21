function [errorClassification,err] = classerror(labels, y)
%CLASSERROR computes the classification error and the classification for each class and averages
%it (class error)
%   
%   Input:
%       labels:      the labels
%       y:   the output of the classifier
%
%   output:
%       errorClassification: the classification error
%       err:    the class-averaged classification error (= Class error)

    classes = unique(labels);
    err_ = zeros(1,length(classes));

    for c=1:length(classes)
        err_(c) = sum((labels~=y) & (labels == classes(c)))./sum(labels==classes(c));% + sum((labels~=y) & (labels == 1))./sum(labels==1));
    end

    err = mean(err_);

    nCorrect = 0;
    labelSize = size(labels);
    for i = 1:labelSize(1)
        if (y(i) == labels(i))
            nCorrect = nCorrect + 1;
        end
    end

    errorClassification = 1 - nCorrect/648;

end