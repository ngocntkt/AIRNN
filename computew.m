function[w] = computew(theta,sigma,regType)
    if(regType == 4)
        w = theta*exp(-theta*sigma);
    end
    if(regType == 2)
        w = 1./(theta + sigma);
    end
end