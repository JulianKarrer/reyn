#ifndef UTILS_DISPATCH_CUH_
#define UTILS_DISPATCH_CUH_

/// @brief Generic higher order function (functor) for choosing an unsigned
/// template parameter at runtime. Selects the smallest integer greater than or
/// equal to the `lower_bound` specified from an ordered list of values. The
/// list of template arguments should be in ascending order, e.g.
/// `RuntimeTemplateSelectList<Launch, 1,2,4,8>`
///
/// - using `dispatch` launches `Launch<N>()` where N from the list is >=
/// `lower_bound` or the last element in the template parameter list (calls the
/// `operator()` of the provided `Launch` Functor, which must have a trivial
/// constructor)
/// @tparam Elem
/// @tparam ...Cons
template <template <unsigned> class Launch, unsigned Elem, unsigned... Cons>
struct RuntimeTemplateSelectList {
    template <typename... Args>
    ///@brief Calls the `operator()` on the `Launch` template parameter with an
    /// unsigned integer as the first template parameter which is the smallest
    /// one greater than or equal to `lower_bound`, from the list of values
    /// passed in as further template arguments.
    ///
    ///@param lower_bound the minimum value for the template argument to select,
    /// from the list of values given
    ///@param args arguments to pass onto the `operator()`
    static void dispatch(unsigned lower_bound, Args&&... args)
    {
        static_assert(std::is_trivially_default_constructible_v<Launch<Elem>>,
            "When using RuntimeTemplateSelectList::dispatch the functor passed "
            "in as the Launch template parameter must be trivially "
            "constructible.");
        if (lower_bound <= Elem) {
            Launch<Elem> {}(std::forward<Args>(args)...);
        } else {
            // pop the first template
            RuntimeTemplateSelectList<Launch, Cons...>::dispatch(
                lower_bound, std::forward<Args>(args)...);
        }
    }
};
/// @brief Base case of the recursively defined `RuntimeTemplateSelectList`
/// template, where the list contains only a single entry, making it equivalent
/// to simply calling `operator()` on the provided `Launch` functor
/// @tparam Last
template <template <unsigned> class Launch, unsigned Last>
struct RuntimeTemplateSelectList<Launch, Last> {
    template <typename... Args>
    static void dispatch(unsigned lower_bound, Args&&... args)
    {
        Launch<Last> {}(std::forward<Args>(args)...);
    }
};

#endif // UTILS_DISPATCH_CUH_