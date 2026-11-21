#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace native_lm {

// Terms return residuals and derivatives with respect to scalar network output.
// The trainer combines them with the network parameter Jacobian.
template <class Target>
class SupervisedMseTerm {
public:
    static constexpr std::size_t kResidualsPerSample = 1;
    static constexpr bool kObjectiveEqualsSupervisedMse = true;

    explicit SupervisedMseTerm(Target target = {}) : target_(std::move(target)) {}

    template <class Sample>
    void evaluate(const Sample& sample, double prediction,
                  double* residuals, double* derivatives_wrt_output) const {
        residuals[0] = prediction - target_(sample);
        derivatives_wrt_output[0] = 1.0;
    }

private:
    Target target_;
};

// Applies sqrt(weight) to a residual block. Consequently the squared
// least-squares objective receives exactly weight * ||residual||^2.
template <class Term>
class ScaledResidualTerm {
public:
    static constexpr std::size_t kResidualsPerSample = Term::kResidualsPerSample;
    static constexpr bool kObjectiveEqualsSupervisedMse = false;

    explicit ScaledResidualTerm(Term term, double weight)
        : term_(std::move(term)), scale_(std::sqrt(weight)) {
        if (!(weight >= 0.0) || !std::isfinite(weight)) {
            throw std::invalid_argument("Residual-term weight must be finite and nonnegative");
        }
    }

    template <class Sample>
    void evaluate(const Sample& sample, double prediction,
                  double* residuals, double* derivatives_wrt_output) const {
        term_.evaluate(sample, prediction, residuals, derivatives_wrt_output);
        for (std::size_t index = 0; index < kResidualsPerSample; ++index) {
            residuals[index] *= scale_;
            derivatives_wrt_output[index] *= scale_;
        }
    }

private:
    Term term_;
    double scale_;
};

// Bound residuals smoothly with unit slope at zero, limiting the influence
// of large cold-start errors.
template <class Term>
class SoftClippedResidualTerm {
public:
    static constexpr std::size_t kResidualsPerSample = Term::kResidualsPerSample;
    static constexpr bool kObjectiveEqualsSupervisedMse = false;

    explicit SoftClippedResidualTerm(Term term, double cap)
        : term_(std::move(term)), cap_(cap) {
        if (!(cap > 0.0) || !std::isfinite(cap)) {
            throw std::invalid_argument("Soft residual cap must be finite and positive");
        }
    }

    template <class Sample>
    void evaluate(const Sample& sample, double prediction,
                  double* residuals, double* derivatives_wrt_output) const {
        term_.evaluate(sample, prediction, residuals, derivatives_wrt_output);
        for (std::size_t index = 0; index < kResidualsPerSample; ++index) {
            const double compressed = std::tanh(residuals[index] / cap_);
            // d[c tanh(r/c)]/dr = 1 - tanh(r/c)^2.
            residuals[index] = cap_ * compressed;
            derivatives_wrt_output[index] *= 1.0 - compressed * compressed;
        }
    }

private:
    Term term_;
    double cap_;
};

// Static composition fixes the residual count used to allocate Jacobian storage.
template <class... Terms>
class StackedResiduals {
public:
    static_assert(sizeof...(Terms) > 0, "An LM objective requires at least one residual term");
    static constexpr std::size_t kResidualsPerSample = (Terms::kResidualsPerSample + ... + 0U);
    static constexpr bool kObjectiveEqualsSupervisedMse =
        (Terms::kObjectiveEqualsSupervisedMse && ...);

    explicit StackedResiduals(Terms... terms) : terms_(std::move(terms)...) {}

    template <class Sample>
    void evaluate(const Sample& sample, double prediction,
                  double* residuals, double* derivatives_wrt_output) const {
        std::size_t offset = 0;
        std::apply([&](const auto&... term) {
            ((term.evaluate(sample, prediction, residuals + offset,
                            derivatives_wrt_output + offset),
              offset += std::decay_t<decltype(term)>::kResidualsPerSample), ...);
        }, terms_);
    }

private:
    std::tuple<Terms...> terms_;
};

}  // namespace native_lm
