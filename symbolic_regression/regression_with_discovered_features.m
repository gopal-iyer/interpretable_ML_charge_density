function rms_e=regression_with_discovered_features(x, A0, b0)

[n,m]=size(A0);

l5 = 1;
l6 = 1;
l7 = 1;
l8 = 1;
l9_1 = 1;
l9_2 = 1;
l10_1 = 1;
l10_2 = 1;
l12_1 = 1;
l12_2 = 1;
l15 = 1;
l16_1 = 1;
l16_2 = 1;
l17 = 1;
l18_1 = 1;
l18_2 = 1;
l19 = 1;
l20 = 1;

l5 = x(1);
l6 = x(2);
l8 = x(3);

clear A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19 A20;

count = 1;

for i1=1:m
  % one feature, polynomials and fractional exponents
  A1(:,count) = sqrt(sqrt(A0(:, i1))); % requires no hyperparameters
  A2(:,count) = sqrt(A0(:, i1)); % requires no hyperparameters
  A3(:,count) = sqrt(A0(:, i1)).^3; % requires no hyperparameters
  A4(:,count) = A0(:, i1).^2; % requires no hyperparameters
  %A4_1(:,count) = A0(:, i1).^3;
  %A4_2(:,count) = A0(:, i1).^4;

  % one feature, exponentials
  A5(:,count) = exp(l5 * A0(:,i1));
  A6(:,count) = exp(-l6 * A0(:,i1).^2);
  A7(:,count) = exp(l7 * A0(:,i1).^2); % rank is often poor !!
  A8(:,count) = exp(l8 * sqrt(A0(:,i1)));
  %A21(:,count) = exp(-l5*A0(:,i1));
  %A22(:,count) = A0(:,i1) .* exp(-l5 * A0(:,i1));
  %A23(:,count) = (A0(:,i1).^2) .* exp(-l5 * A0(:,i1));
  %A24(:,count) = (A0(:,i1).^3) .* exp(-l5 * A0(:,i1));

  count = count + 1;
end

%{
count = 1;
for i1=1:m
  for i2=i1:m
     % two features, polynomials and fractional exponents
     A9(:,count)  = sqrt(l9_1 * A0(:,i1) + l9_2 * A0(:,i2));
     A10(:,count) = sqrt((l10_1 * A0(:,i1) + l10_2 * A0(:,i2)).^3);
     A11(:,count) = A0(:,i1) .* sqrt(A0(:,i2)); % requires no hyperparameters
     A12(:,count) = sqrt((l12_1 * A0(:,i1).^2 + l12_2 * A0(:,i2)).^3);
     A13(:,count) = (A0(:,i1).^2).*A0(:,i2); % requires no hyperparameters
     A14(:,count) = (A0(:,i1).^2).*(A0(:,i2).^2); % requires no hyperparameter

     % two features, exponentials
     A15(:,count) = A0(:,i1) .* exp(l15 * A0(:,i2));
     A16(:,count) = A0(:,i1) .* exp(l16_1 * exp(-l16_2 * A0(:,i2)));
     A17(:,count) = exp(-l17 * A0(:,i1).* A0(:,i2));
     A18(:,count) = exp(-(l18_1 * A0(:,i1) + l18_2 * A0(:,i2)));
     A19(:,count) = (A0(:,i1) .* A0(:,i2)) .* exp(-l19 * A0(:,i2).^2);
     A20(:,count) = (A0(:,i1).*A0(:,i2)) .* exp(l20 * A0(:,i2));

     %A25(:,count) = A0(:,i1) .* exp(-l5 * A0(:,i2));
     %A26(:,count) = (A0(:,i1).^2) .* exp(-l5 * A0(:,i2));
     %A27(:,count) = (A0(:,i1).^3) .* exp(-l5 * A0(:,i1));

     %A28(:,count) = (A0(:,i1).*A0(:,i2)) .* exp(-l5 * A0(:,i2));
     %A29(:,count) = ((A0(:,i1).^2).*A0(:,i2)) .* exp(-l5 * A0(:,i2));
     %A30(:,count) = ((A0(:,i1).^3).*A0(:,i2)) .* exp(-l5 * A0(:,i2));

     %A31(:,count) = (A0(:,i1).*(A0(:,i2).^2)) .* exp(-l5 * A0(:,i2));
     %A32(:,count) = ((A0(:,i1).^2).*(A0(:,i2).^2)) .* exp(-l5 * A0(:,i2));
     %A33(:,count) = ((A0(:,i1).^3).*(A0(:,i2).^2)) .* exp(-l5 * A0(:,i2));

     %A34(:,count) = (A0(:,i1).*(A0(:,i2).^3)) .* exp(-l5 * A0(:,i2));
     %A35(:,count) = ((A0(:,i1).^2).*(A0(:,i2).^3)) .* exp(-l5 * A0(:,i2));
     %A36(:,count) = ((A0(:,i1).^3).*(A0(:,i2).^3)) .* exp(-l5 * A0(:,i2));

     count = count + 1;
  end
end
%}

A = [A0 A1 A2 A3 A4 A5 A6 A8];

x0 = A\b0;
x_solved = x0;
%load('solved_coeffs_nb_6p2.mat');
r0=A*x_solved-b0;
rms_e = sqrt(mean(r0.^2));
d2 = ['rmse    = ', num2str(rms_e)];
disp(d2);
l2_score=sum(r0.^2);
l5
l6
l8
save('solved_coeffs_nb_6p2', 'x_solved');
disp('-----------------------------');
