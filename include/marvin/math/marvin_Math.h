// ========================================================================================================
//  _______ _______ ______ ___ ___ _______ _______
// |   |   |   _   |   __ \   |   |_     _|    |  |
// |       |       |      <   |   |_|   |_|       |
// |__|_|__|___|___|___|__|\_____/|_______|__|____|
//
// This file is part of the Marvin open source library and is licensed under the terms of the MIT License.
//
// ========================================================================================================

#ifndef MARVIN_MATH_H
#define MARVIN_MATH_H
#include "marvin/library/marvin_Concepts.h"
#include "marvin/utils/marvin_Range.h"
#include <cmath>
#include <algorithm>
#include <span>
#include <complex>
#include <cassert>
#include <numbers>

namespace marvin::math {

    /**
        Returns a point `ratio` of the way between `start` and `end`.
        \param start The start value of the interpolation.
        \param end The end value of the interpolation.
        \param ratio The (0 to 1) position between `start` and `end`
        \return A linear interpolation between `start` and `end` at position `ratio`.
    */
    template <FloatType T>
    [[nodiscard]] T lerp(T start, T end, T ratio) noexcept {
        const auto interpolated = start + (end - start) * ratio;
        return interpolated;
    }


    /**
     * Takes a value in range 0 to 1, and rescales it to be in range `newMin` to `newMax`.
     * \param x The value to remap.
     * \param newMin The new range's min.
     * \param newMax The new range's max.
     * \return The rescaled value.
     */
    template <FloatType T>
    [[nodiscard]] T remap(T x, T newMin, T newMax) {
        // (v_n . (mx - mn)) + mn = v
        const auto rescaled = (x * (newMax - newMin)) + newMin;
        return rescaled;
    }

    /**
        Takes a value in range `srcMin` to `srcMax`, normalises it, and rescales it to be in range
        `newMin` to `newMax`
        \param x The value to remap.
        \param srcMin The min of the value's original range.
        \param srcMax The max of the value's original range.
        \param newMin The new range's min.
        \param newMax The new range's max.
        \return The rescaled value.
    */
    template <FloatType T>
    [[nodiscard]] T remap(T x, T srcMin, T srcMax, T newMin, T newMax) {
        // (v - mn) / (mx - mn) == v_n
        const auto normalised = (x - srcMin) / (srcMax - srcMin);
        return remap<T>(normalised, newMin, newMax);
    }

    /**
     * Takes a value in range `srcRange`, normalises it, and remaps it to be in range `newRange`.
     * \param x The value to remap.
     * \param srcRange The range the value is currently in.
     * \param newRange The range to map the value into.
     * \return The rescaled value.
     */
    template <FloatType T>
    [[nodiscard]] T remap(T x, marvin::utils::Range<T> srcRange, marvin::utils::Range<T> newRange) {
        return remap<T>(x, srcRange.min, srcRange.max, newRange.min, newRange.max);
    }

    /**
     * Takes a value in range 0 to 1, and rescales it to be in range `newRange`.
     * \param x The value to remap.
     * \param newRange The range to map the value into.
     * \return The rescaled value.
     */
    template <FloatType T>
    [[nodiscard]] T remap(T x, marvin::utils::Range<T> newRange) {
        return remap<T>(x, newRange.min, newRange.max);
    }

    /**
     * Finds the RMS (root mean square) of an array-like of floating point values. In the case of a 0-sized array, 0
     * be returned.
     * \param data The data to find the RMS of.
     * \return The RMS of the data given.
     */
    template<FloatType T>
    [[nodiscard]] T rms(std::span<T> data) {
        if (data.empty()) {
            return static_cast<T>(0);
        }

        const auto sum = std::accumulate(data.begin(), data.end(), static_cast<T>(0), [] (auto acc, auto next) {
            return acc + next * next;
        });
        const auto mean = sum / static_cast<T>(data.size());
        return std::sqrt(mean);
    }

    /**
     * Finds the RMS (root mean square) of a multidimensional array of floating point values. Can be used to find the
     * RMS of stereo audio as a single value. In such a case, the combined RMS is calculated as:
     *
     * ```
     * RMS_stereo = sqrt((RMS_l^2 + RMS_r^2) / 2)
     * ```
     *
     * In the case of a 0-sized outer array, 0 is returned. If the size of one of the inner arrays is 0, 0 is used in
     * the calculation for that array.
     *
     * \param data An array-like of array-likes of floating point values to find the RMS of.
     * \return The combined RMS of all data.
     */
    template<FloatType T>
    [[nodiscard]] T rms(std::span<std::span<T>> data) {
        if (data.empty()) {
            return static_cast<T>(0);
        }

        const auto getChannelMean = [] (std::span<T> channel) -> T {
            if (channel.empty()) {
                return static_cast<T>(0);
            }

            const auto sum = std::accumulate(channel.begin(), channel.end(), static_cast<T>(0), [] (auto acc, auto next) {
                return acc + next * next;
            });
            return sum / static_cast<T>(channel.size());
        };

        const auto sum = std::accumulate(data.begin(), data.end(), static_cast<T>(0), [&] (auto acc, const auto& next) {
            return acc + getChannelMean(next);
        });
        const auto mean = sum / static_cast<T>(data.size());
        return std::sqrt(mean);
    }

    /**
        Takes an array-like of complexes represented as `[re, im, re, im, ... im]` of size `N` and creates
        a non-owning view treating the interleaved stream as an array-like of type `std::complex<T>` `N/2` points long,
        with no copies or allocations.<br>
        Uses terrifying reinterpret-cast shennanigans to do so, and <b>absolutely requires the caller to ensure that
        `data.size()` is even<b>.
        \param data The data to create a view into.
        \return A non owning view treating data as an array-like<std::complex<T>> `data.size() / 2` points long.
    */
    template <FloatType T>
    [[nodiscard]] std::span<std::complex<T>> interleavedToComplexView(std::span<T> data) {
        assert(data.size() % 2 == 0);
        std::span<std::complex<T>> complexView{ reinterpret_cast<std::complex<T>*>(data.data()), data.size() / 2 };
        return complexView;
    }

    /**
        Takes an array-like of `std::complex<T>`s of size `N`, and creates a non owning view treating the packed complex data as an interleaved
        stream of reals, represented as `[re, im, re, im... im]` of size `N*2`.
        Uses terrifying reinterpret-cast shennanigans to do so, and <b>absolutely requires the caller to ensure that
        `data.size()` is even<b>.
        \param data The data to create the view into.
        \return A non owning view treating the data as an interleaved stream of reals `data.size() * 2` points long.
     */
    template <FloatType T>
    [[nodiscard]] std::span<T> complexViewToInterleaved(std::span<std::complex<T>> data) {
        std::span<T> interleavedView{ reinterpret_cast<T*>(data.data()), data.size() * 2 };
        return interleavedView;
    }
    template <FloatType T>
    [[nodiscard]] T sinc(T x) noexcept {
        static constexpr auto epsilon{ static_cast<T>(1e-6) };
        if (std::abs(x) < epsilon) {
            return static_cast<T>(1.0);
        }
        const auto xPi = x * std::numbers::pi_v<T>;
        return std::sin(xPi) / xPi;
    }

} // namespace marvin::math

#endif
