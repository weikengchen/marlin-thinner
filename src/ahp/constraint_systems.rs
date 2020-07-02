#![allow(non_snake_case)]

use crate::ahp::indexer::Matrix;
use crate::ahp::*;
use crate::{BTreeMap, BinaryHeap, Cow, String, ToString};
use algebra_core::{Field, PrimeField};
use core::cmp::{max, min, Ordering};
use derivative::Derivative;
use ff_fft::{
    cfg_iter_mut, EvaluationDomain, Evaluations as EvaluationsOnDomain, GeneralEvaluationDomain,
};
use poly_commit::LabeledPolynomial;
use r1cs_core::{ConstraintSystem, Index as VarIndex, LinearCombination, SynthesisError, Variable};

/* ************************************************************************* */
/* ************************************************************************* */
/* ************************************************************************* */

// This function converts a matrix output by Zexe's constraint infrastructure
// to the one used in this crate.
fn to_matrix_helper<F: Field>(
    matrix: &[Vec<(F, VarIndex)>],
    num_input_variables: usize,
) -> Matrix<F> {
    let mut new_matrix = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut new_row = Vec::with_capacity(row.len());
        for (fe, column) in row {
            let column = match column {
                VarIndex::Input(i) => *i,
                VarIndex::Aux(i) => num_input_variables + i,
            };
            new_row.push((*fe, column))
        }
        new_matrix.push(new_row)
    }
    new_matrix
}

fn balance_matrices<F: Field>(
    num_input_variables: usize,
    num_witness_variables: &mut usize,
    num_constraints: &mut usize,
    additional_balancing_constraints: &mut Matrix<F>,
    a_matrix: &mut Matrix<F>,
    b_matrix: &mut Matrix<F>,
    c_matrix: &mut Matrix<F>,
    balance_target: usize,
) {
    // Step 1: define the heap entry data type
    #[cfg_attr(rustfmt, rustfmt_skip)]
    struct IndexedEntrySize { index: usize, is_AB: bool, value: usize, other_value_if_AB: usize };
    #[cfg_attr(rustfmt, rustfmt_skip)]
    impl IndexedEntrySize {
        fn new(index: usize, is_AB: bool, value: usize, other_value_if_AB: usize) -> Self { Self {  index, is_AB, value, other_value_if_AB } }
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    impl PartialOrd for IndexedEntrySize {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            if self.is_AB == true {
                Some(self.value.cmp(&other.value).then(other.other_value_if_AB.cmp(&self.other_value_if_AB))) // prefer the one with the biggest gap between A/B
            } else {
                Some(self.value.cmp(&other.value))
            }
        }
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    impl PartialEq for IndexedEntrySize {
        fn eq(&self, other: &Self) -> bool {
            if self.is_AB == true {
                self.value.eq(&other.value) && self.other_value_if_AB.eq(&other.other_value_if_AB)
            } else {
                self.value.eq(&other.value)
            }
        }
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    impl Ord for IndexedEntrySize {
        fn cmp(&self, other: &Self) -> Ordering {
            if self.is_AB == true {
                self.value.cmp(&other.value).then(other.other_value_if_AB.cmp(&self.other_value_if_AB))
            } else {
                self.value.cmp(&other.value)
            }
        }
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    impl Eq for IndexedEntrySize {}

    // Step 2: compute the current density
    let mut current_a_density: usize = a_matrix.iter().map(|row| row.len()).sum();
    let mut current_b_density: usize = b_matrix.iter().map(|row| row.len()).sum();
    let mut current_c_density: usize = c_matrix.iter().map(|row| row.len()).sum();
    let mut current_overall_density =
        max(max(current_a_density, current_b_density), current_c_density);

    // Step 3: initialize the counters for new constraints
    let mut num_new_witness_variables = 0usize;
    let witness_index_offset = num_input_variables + *num_witness_variables;

    // Step 4: build the data structure of triple heaps, in which `deletions` are virtually synchronized by the `deleted_constraints` array
    let mut deleted_constraints = vec![false; a_matrix.len()];
    let (mut a_heap, mut b_heap, mut c_heap) = {
        let mut indexed_a_matrix = Vec::<IndexedEntrySize>::new();
        let mut indexed_b_matrix = Vec::<IndexedEntrySize>::new();
        let mut indexed_c_matrix = Vec::<IndexedEntrySize>::new();
        #[cfg_attr(rustfmt, rustfmt_skip)]
        for (i, ((a, b), c)) in a_matrix.iter().zip(b_matrix.iter()).zip(c_matrix.iter()).enumerate()
        {
            indexed_a_matrix.push(IndexedEntrySize::new(i, true, a.len(), b.len()));
            indexed_b_matrix.push(IndexedEntrySize::new(i, true, b.len(), a.len()));
            indexed_c_matrix.push(IndexedEntrySize::new(i, true, c.len(), 0));
        }
        #[cfg_attr(rustfmt, rustfmt_skip)]
        (BinaryHeap::from(indexed_a_matrix), BinaryHeap::from(indexed_b_matrix), BinaryHeap::from(indexed_c_matrix))
    };

    // Step 5: start the loop for reducing the densest matrix; by default, swappable is used
    let mut do_swappable_if_possible = true;
    loop {
        // Step 5.1: if the density is close to the target, stop
        if balance_target != 0
            && current_overall_density
                - min(min(current_a_density, current_b_density), current_c_density)
                <= balance_target
        {
            break;
        }

        // Step 5.2: find the densest matrix
        let is_a_densest = current_a_density == current_overall_density;
        let is_b_densest = (!is_a_densest) && (current_b_density == current_overall_density);

        // Step 5.3: get the densest LC in this densest matrix
        let mut densest_lc_index: usize;
        let mut densest_lc_count: usize;
        loop {
            // take the longest LC from the max heap of that matrix.
            #[cfg_attr(rustfmt, rustfmt_skip)]
            let res = if is_a_densest {  a_heap.pop() } else if is_b_densest { b_heap.pop() } else { c_heap.pop() }.unwrap();

            densest_lc_index = res.index;
            densest_lc_count = res.value;

            // require this densest LC to be existing, not a deleted one (this is part of the synchronization of the three heaps)
            if deleted_constraints[densest_lc_index] == false {
                break;
            }
        }

        // Step 5.4: obtain the constraint that has this densest LC
        let a = &a_matrix[densest_lc_index];
        let b = &b_matrix[densest_lc_index];
        let c = &c_matrix[densest_lc_index];
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let cur = if is_a_densest { a } else if is_b_densest { b } else { c };

        // Step 5.5: initialize the data structures for storing the suggestion.
        let mut suggested_constraints =
            Vec::<(Vec<(F, usize)>, Vec<(F, usize)>, Vec<(F, usize)>)>::new();
        let mut suggested_new_variables = Vec::<Vec<(F, usize)>>::new();

        macro_rules! add_new_witness_variable {
            // `()` indicates that the macro takes no argument.
            ($index:ident, $variable:ident, $copy_from:expr) => {
                let $index = witness_index_offset
                    + num_new_witness_variables
                    + suggested_new_variables.len();
                let $variable = $copy_from.clone();

                suggested_new_variables.push($variable.clone());
            };
        }

        // Step 5.6: Discuss case by case
        if is_a_densest || is_b_densest {
            // Case 1: a or b is the densest
            //   There are two possible strategies:
            //      (1) swap a and b, if the other one is smaller; it has the benefit of no new constraint but it never pushes things to c
            //      (2) relocate or split this a or b

            if do_swappable_if_possible
                && ((is_a_densest && a.len() > b.len()) || (is_b_densest && b.len() > a.len()))
            {
                // Case 1.(1): swap
                suggested_constraints.push((b.clone(), a.clone(), c.clone()));
            } else {
                // Case 1.(2): relocate or split
                let the_other_ab_density = if is_a_densest {
                    current_b_density
                } else {
                    current_a_density
                };
                let diff = max(the_other_ab_density, current_c_density)
                    - min(the_other_ab_density, current_c_density);
                if diff >= densest_lc_count {
                    // relocate
                    add_new_witness_variable!(u_index, u, cur);
                    if the_other_ab_density < current_c_density {
                        if is_a_densest {
                            // rewrite A * B = C to
                            //      U * B = C
                            //      1 * A = U
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), u_index)],  b.clone(), c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), 0)], u.clone(), vec![(F::one(), u_index)]));
                        } else {
                            // rewrite A * B = C to
                            //      A * U = C
                            //      B * 1 = U
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((a.clone(), vec![(F::one(), u_index)], c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((u.clone(), vec![(F::one(), 0)], vec![(F::one(), u_index)]));
                        }
                    } else {
                        if is_a_densest {
                            // rewrite A * B = C to
                            //      U * B = C
                            //      U * 1 = A
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), u_index)], b.clone(), c.clone()));
                        } else {
                            // rewrite A * B = C to
                            //      A * U = C
                            //      U * 1 = B
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((a.clone(), vec![(F::one(), u_index)], c.clone()));
                        }
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                         suggested_constraints.push((vec![(F::one(), u_index)], vec![(F::one(), 0)], u.clone()));
                    }
                } else {
                    // split
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    add_new_witness_variable!(u_index, u, cur[..(densest_lc_count - diff) / 2].to_vec());
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    add_new_witness_variable!(v_index, v, cur[(densest_lc_count - diff) / 2..].to_vec());

                    if the_other_ab_density < current_c_density {
                        // c has less space, so give c the lighter one (which is u)
                        if is_a_densest {
                            // rewrite A * B = C to
                            //      (U + V) * B = C
                            //      1 * A_2 = V
                            //      1 * U = A_1
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), u_index), (F::one(), v_index)], b.clone(), c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), 0)], v.clone(), vec![(F::one(), v_index)]));
                        } else {
                            // rewrite A * B = C to
                            //      A * (U + V) = C
                            //      B_2 * 1 = V
                            //      1 * U = B_1
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((a.clone(), vec![(F::one(), u_index), (F::one(), v_index)], c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((v.clone(), vec![(F::one(), 0)], vec![(F::one(), v_index)]));
                        }
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        suggested_constraints.push((vec![(F::one(), 0)], vec![(F::one(), u_index)], u.clone()));
                    } else {
                        // c has more space, so give c the heavier one (which is v)
                        if is_a_densest {
                            // rewrite A * B = C to
                            //      (U + V) * B = C
                            //      1 * A_2 = U
                            //      1 * V = A_1
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), u_index), (F::one(), v_index)], b.clone(), c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), 0)], u.clone(), vec![(F::one(), u_index)]));
                        } else {
                            // rewrite A * B = C to
                            //      A * (U + V) = C
                            //      B_2 * 1 = U
                            //      1 * V = B_1
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((a.clone(), vec![(F::one(), u_index), (F::one(), v_index)], c.clone()));
                            #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((u.clone(), vec![(F::one(), 0)], vec![(F::one(), u_index)]));
                        }
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        suggested_constraints.push((vec![(F::one(), 0)], vec![(F::one(), v_index)], v.clone()));
                    }
                }
            }
        } else {
            // Case 2: c is the densest; strategy: relocate or split
            let diff = max(current_a_density, current_b_density)
                - min(current_a_density, current_c_density);
            if diff >= densest_lc_count {
                // relocate
                if a.len() == 0 && b.len() == 0 {
                    // the case of 0 * 0 = LC
                    if current_a_density < current_b_density {
                        // rewrite 0 * 0 = C to
                        //      C * 1 = 0
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        suggested_constraints.push((c.clone(), vec![(F::one(), 0)], Vec::new()));
                    } else {
                        // rewrite 0 * 0 = C to
                        //      1 * C = 0
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        suggested_constraints.push((vec![(F::one(), 0)], c.clone(), Vec::new()));
                    }
                } else {
                    add_new_witness_variable!(u_index, u, c);
                    if current_a_density < current_b_density {
                        // rewrite A * B = C to
                        //      C * 1 = U
                        //      A * B = U
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((c.clone(), vec![(F::one(), 0)], vec![(F::one(), u_index)]));
                    } else {
                        // rewrite A * B = C to
                        //      1 * C = U
                        //      A * B = U
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                            suggested_constraints.push((vec![(F::one(), 0)], c.clone(), vec![(F::one(), u_index)]));
                    }
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                        suggested_constraints.push((a.clone(), b.clone(), vec![(F::one(), u_index)]));
                }
            } else {
                // split
                add_new_witness_variable!(
                    u_index,
                    u,
                    cur[..(densest_lc_count - diff) / 2].to_vec()
                );
                add_new_witness_variable!(
                    v_index,
                    v,
                    cur[(densest_lc_count - diff) / 2..].to_vec()
                );

                if current_a_density < current_b_density {
                    // rewrite A * B = C to
                    //      1 * C_1 = U
                    //      C_2 * 1 = V
                    //      A * B = (U + V)
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    suggested_constraints.push((vec![(F::one(), 0)], u.clone(), vec![(F::one(), u_index)]));
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    suggested_constraints.push((v.clone(), vec![(F::one(), 0)], vec![(F::one(), v_index)]));
                } else {
                    // rewrite A * B = C to
                    //      1 * C_1 = V
                    //      C_2 * 1 = U
                    //      A * B = (U + V)
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    suggested_constraints.push((vec![(F::one(), 0)], v.clone(), vec![(F::one(), v_index)]));
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    suggested_constraints.push((u.clone(), vec![(F::one(), 0)], vec![(F::one(), u_index)]));
                }
                #[cfg_attr(rustfmt, rustfmt_skip)]
                suggested_constraints.push((a.clone(), b.clone(), vec![(F::one(), u_index), (F::one(), v_index)]));
            }
        }

        // Step 5.7: compute the new density if the suggested changes are accepted
        let new_a_density = current_a_density - a.len()
            + suggested_constraints
                .iter()
                .map(|x| x.0.len())
                .sum::<usize>();
        let new_b_density = current_b_density - b.len()
            + suggested_constraints
                .iter()
                .map(|x| x.1.len())
                .sum::<usize>();
        let new_c_density = current_c_density - c.len()
            + suggested_constraints
                .iter()
                .map(|x| x.2.len())
                .sum::<usize>();
        let new_overall_density = max(max(new_a_density, new_b_density), new_c_density);

        // Step 5.8: compute if there is improvement
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let current_penalty =
            ((current_a_density * current_a_density + current_b_density * current_b_density + current_c_density * current_c_density) as f64).sqrt();
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let new_penalty =
            ((new_a_density * new_a_density + new_b_density * new_b_density + new_c_density * new_c_density) as f64).sqrt();

        // Step 5.9: if there is no improvement and swapping has been used, stop swapping
        if new_penalty >= current_penalty && do_swappable_if_possible {
            do_swappable_if_possible = false;
            continue;
        }

        // Step 5.10: stop when there is no improvement
        if do_swappable_if_possible == false && new_penalty >= current_penalty {
            break;
        }

        // Step 5.11: now, we are going to take the suggestion

        // delete the constraint
        deleted_constraints[densest_lc_index] = true;

        // update the density
        current_a_density = new_a_density;
        current_b_density = new_b_density;
        current_c_density = new_c_density;
        current_overall_density = new_overall_density;

        // insert the new variables
        num_new_witness_variables += suggested_new_variables.len();
        additional_balancing_constraints.append(&mut suggested_new_variables);

        // insert the new constraints to the heap
        let mut index = a_matrix.len();
        for (a, b, c) in suggested_constraints.iter() {
            a_heap.push(IndexedEntrySize::new(index, true, a.len(), b.len()));
            b_heap.push(IndexedEntrySize::new(index, true, b.len(), a.len()));
            c_heap.push(IndexedEntrySize::new(index, false, c.len(), 0));
            index += 1;
        }

        // insert the new constraints to the matrix
        for (a, b, c) in suggested_constraints.iter() {
            a_matrix.push(a.clone());
            b_matrix.push(b.clone());
            c_matrix.push(c.clone());
        }

        // add placeholders in the `deleted_constraints`
        for _ in suggested_constraints.iter() {
            deleted_constraints.push(false);
        }
    }

    // Step 6: save the result

    // update the number of witness variables in the CS
    *num_witness_variables += num_new_witness_variables;

    // remove the deleted constraints from a, b, c
    *a_matrix = a_matrix
        .iter()
        .enumerate()
        .filter(|(i, _)| deleted_constraints[*i] == false)
        .map(|(_, x)| x.clone())
        .collect();
    *b_matrix = b_matrix
        .iter()
        .enumerate()
        .filter(|(i, _)| deleted_constraints[*i] == false)
        .map(|(_, x)| x.clone())
        .collect();
    *c_matrix = c_matrix
        .iter()
        .enumerate()
        .filter(|(i, _)| deleted_constraints[*i] == false)
        .map(|(_, x)| x.clone())
        .collect();

    // update the number of constraints (after the deletion)
    *num_constraints = a_matrix.len();
}

/// Stores constraints during index generation.
pub(crate) struct IndexerConstraintSystem<F: Field> {
    pub(crate) num_input_variables: usize,
    pub(crate) num_witness_variables: usize,
    pub(crate) num_constraints: usize,

    pub(crate) additional_balancing_constraints: Matrix<F>,
    pub(crate) a: Vec<Vec<(F, VarIndex)>>,
    pub(crate) b: Vec<Vec<(F, VarIndex)>>,
    pub(crate) c: Vec<Vec<(F, VarIndex)>>,

    pub(crate) a_matrix: Option<Matrix<F>>,
    pub(crate) b_matrix: Option<Matrix<F>>,
    pub(crate) c_matrix: Option<Matrix<F>>,
}

impl<F: Field> IndexerConstraintSystem<F> {
    pub(crate) fn process_matrices(&mut self) {
        let mut a = to_matrix_helper(&self.a, self.num_input_variables);
        let mut b = to_matrix_helper(&self.b, self.num_input_variables);
        let mut c = to_matrix_helper(&self.c, self.num_input_variables);

        let a_density: usize = a.iter().map(|row| row.len()).sum();
        let b_density: usize = b.iter().map(|row| row.len()).sum();
        let c_density: usize = c.iter().map(|row| row.len()).sum();

        balance_matrices(
            self.num_input_variables,
            &mut self.num_witness_variables,
            &mut self.num_constraints,
            &mut self.additional_balancing_constraints,
            &mut a,
            &mut b,
            &mut c,
            ((a_density + b_density + c_density) as f64 / 3.0 * 0.1) as usize,
        );

        self.a_matrix = Some(a);
        self.b_matrix = Some(b);
        self.c_matrix = Some(c);
    }

    #[inline]
    fn make_row(l: &LinearCombination<F>) -> Vec<(F, VarIndex)> {
        l.as_ref()
            .iter()
            .map(|(var, coeff)| (*coeff, var.get_unchecked()))
            .collect()
    }

    pub(crate) fn new() -> Self {
        Self {
            num_input_variables: 1,
            num_witness_variables: 0,
            num_constraints: 0,
            additional_balancing_constraints: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            a_matrix: None,
            b_matrix: None,
            c_matrix: None,
        }
    }

    pub(crate) fn constraint_matrices(self) -> Option<(Matrix<F>, Matrix<F>, Matrix<F>)> {
        let (a, b, c) = (self.a_matrix, self.b_matrix, self.c_matrix);
        match (a, b, c) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        }
    }

    pub(crate) fn num_non_zero(&self) -> usize {
        let a_density = self
            .a_matrix
            .as_ref()
            .unwrap()
            .iter()
            .map(|row| row.len())
            .sum();
        let b_density = self
            .b_matrix
            .as_ref()
            .unwrap()
            .iter()
            .map(|row| row.len())
            .sum();
        let c_density = self
            .c_matrix
            .as_ref()
            .unwrap()
            .iter()
            .map(|row| row.len())
            .sum();

        let max = *[a_density, b_density, c_density]
            .iter()
            .max()
            .expect("iterator is not empty");
        max
    }

    pub(crate) fn make_matrices_square(&mut self) {
        let num_variables = self.num_input_variables + self.num_witness_variables;
        let matrix_dim = padded_matrix_dim(num_variables, self.num_constraints);
        make_matrices_square(self, num_variables);
        assert_eq!(
            self.num_input_variables + self.num_witness_variables,
            self.num_constraints,
            "padding failed!"
        );
        assert_eq!(
            self.num_input_variables + self.num_witness_variables,
            matrix_dim,
            "padding does not result in expected matrix size!"
        );
    }
}

impl<ConstraintF: Field> ConstraintSystem<ConstraintF> for IndexerConstraintSystem<ConstraintF> {
    type Root = Self;

    #[inline]
    fn alloc<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<ConstraintF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // There is no assignment, so we don't invoke the
        // function for obtaining one.

        let index = self.num_witness_variables;
        self.num_witness_variables += 1;

        Ok(Variable::new_unchecked(VarIndex::Aux(index)))
    }

    #[inline]
    fn alloc_input<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<ConstraintF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // There is no assignment, so we don't invoke the
        // function for obtaining one.

        let index = self.num_input_variables;
        self.num_input_variables += 1;

        Ok(Variable::new_unchecked(VarIndex::Input(index)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
        LB: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
        LC: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
    {
        self.a.push(Self::make_row(&a(LinearCombination::zero())));
        self.b.push(Self::make_row(&b(LinearCombination::zero())));
        self.c.push(Self::make_row(&c(LinearCombination::zero())));

        self.num_constraints += 1;
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn num_constraints(&self) -> usize {
        self.num_constraints
    }
}

/// This must *always* be in sync with `make_matrices_square`.
pub(crate) fn padded_matrix_dim(num_formatted_variables: usize, num_constraints: usize) -> usize {
    max(num_formatted_variables, num_constraints)
}

pub(crate) fn make_matrices_square<F: Field, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    num_formatted_variables: usize,
) {
    let num_constraints = cs.num_constraints();
    let matrix_padding = ((num_formatted_variables as isize) - (num_constraints as isize)).abs();

    if num_formatted_variables > num_constraints {
        use core::convert::identity as iden;
        // Add dummy constraints of the form 0 * 0 == 0
        for i in 0..matrix_padding {
            cs.enforce(|| format!("pad constraint {}", i), iden, iden, iden);
        }
    } else {
        // Add dummy unconstrained variables
        for i in 0..matrix_padding {
            let _ = cs
                .alloc(|| format!("pad var {}", i), || Ok(F::one()))
                .expect("alloc failed");
        }
    }
}

#[derive(Derivative)]
#[derivative(Clone(bound = "F: PrimeField"))]
pub struct MatrixEvals<'a, F: PrimeField> {
    /// Evaluations of the LDE of row.
    pub row: Cow<'a, EvaluationsOnDomain<F>>,
    /// Evaluations of the LDE of col.
    pub col: Cow<'a, EvaluationsOnDomain<F>>,
    /// Evaluations of the LDE of val.
    pub val: Cow<'a, EvaluationsOnDomain<F>>,
}

/// Contains information about the arithmetization of the matrix M^*.
/// Here `M^*(i, j) := M(j, i) * u_H(j, j)`. For more details, see [COS19].
#[derive(Derivative)]
#[derivative(Clone(bound = "F: PrimeField"))]
pub struct MatrixArithmetization<'a, F: PrimeField> {
    /// LDE of the row indices of M^*.
    pub row: LabeledPolynomial<'a, F>,
    /// LDE of the column indices of M^*.
    pub col: LabeledPolynomial<'a, F>,
    /// LDE of the non-zero entries of M^*.
    pub val: LabeledPolynomial<'a, F>,
    /// LDE of the vector containing entry-wise products of `row` and `col`,
    /// where `row` and `col` are as above.
    pub row_col: LabeledPolynomial<'a, F>,

    /// Evaluation of `self.row`, `self.col`, and `self.val` on the domain `K`.
    pub evals_on_K: MatrixEvals<'a, F>,

    /// Evaluation of `self.row`, `self.col`, and, `self.val` on
    /// an extended domain B (of size > `3K`).
    // TODO: rename B everywhere.
    pub evals_on_B: MatrixEvals<'a, F>,

    /// Evaluation of `self.row_col` on an extended domain B (of size > `3K`).
    pub row_col_evals_on_B: Cow<'a, EvaluationsOnDomain<F>>,
}

// TODO for debugging: add test that checks result of arithmetize_matrix(M).
pub(crate) fn arithmetize_matrix<'a, F: PrimeField>(
    matrix_name: &str,
    matrix: &mut Matrix<F>,
    interpolation_domain: GeneralEvaluationDomain<F>,
    output_domain: GeneralEvaluationDomain<F>,
    input_domain: GeneralEvaluationDomain<F>,
    expanded_domain: GeneralEvaluationDomain<F>,
) -> MatrixArithmetization<'a, F> {
    let matrix_time = start_timer!(|| "Computing row, col, and val LDEs");

    let elems: Vec<_> = output_domain.elements().collect();

    let mut row_vec = Vec::new();
    let mut col_vec = Vec::new();
    let mut val_vec = Vec::new();

    let eq_poly_vals_time = start_timer!(|| "Precomputing eq_poly_vals");
    let eq_poly_vals: BTreeMap<F, F> = output_domain
        .elements()
        .zip(output_domain.batch_eval_unnormalized_bivariate_lagrange_poly_with_same_inputs())
        .collect();
    end_timer!(eq_poly_vals_time);

    let lde_evals_time = start_timer!(|| "Computing row, col and val evals");
    let mut inverses = Vec::new();

    let mut count = 0;

    // Recall that we are computing the arithmetization of M^*,
    // where `M^*(i, j) := M(j, i) * u_H(j, j)`.
    for (r, row) in matrix.into_iter().enumerate() {
        if !is_in_ascending_order(&row, |(_, a), (_, b)| a < b) {
            row.sort_by(|(_, a), (_, b)| a.cmp(b));
        };

        for &mut (val, i) in row {
            let row_val = elems[r];
            let col_val = elems[output_domain.reindex_by_subdomain(input_domain, i)];

            // We are dealing with the transpose of M
            row_vec.push(col_val);
            col_vec.push(row_val);
            val_vec.push(val);
            inverses.push(eq_poly_vals[&col_val]);

            count += 1;
        }
    }
    algebra_core::fields::batch_inversion::<F>(&mut inverses);

    cfg_iter_mut!(val_vec)
        .zip(inverses)
        .for_each(|(v, inv)| *v *= &inv);
    end_timer!(lde_evals_time);

    for _ in 0..(interpolation_domain.size() - count) {
        col_vec.push(elems[0]);
        row_vec.push(elems[0]);
        val_vec.push(F::zero());
    }
    let row_col_vec: Vec<_> = row_vec
        .iter()
        .zip(&col_vec)
        .map(|(row, col)| *row * col)
        .collect();

    let interpolate_time = start_timer!(|| "Interpolating on K and B");
    let row_evals_on_K = EvaluationsOnDomain::from_vec_and_domain(row_vec, interpolation_domain);
    let col_evals_on_K = EvaluationsOnDomain::from_vec_and_domain(col_vec, interpolation_domain);
    let val_evals_on_K = EvaluationsOnDomain::from_vec_and_domain(val_vec, interpolation_domain);
    let row_col_evals_on_K =
        EvaluationsOnDomain::from_vec_and_domain(row_col_vec, interpolation_domain);

    let row = row_evals_on_K.clone().interpolate();
    let col = col_evals_on_K.clone().interpolate();
    let val = val_evals_on_K.clone().interpolate();
    let row_col = row_col_evals_on_K.interpolate();

    let row_evals_on_B =
        EvaluationsOnDomain::from_vec_and_domain(expanded_domain.fft(&row), expanded_domain);
    let col_evals_on_B =
        EvaluationsOnDomain::from_vec_and_domain(expanded_domain.fft(&col), expanded_domain);
    let val_evals_on_B =
        EvaluationsOnDomain::from_vec_and_domain(expanded_domain.fft(&val), expanded_domain);
    let row_col_evals_on_B =
        EvaluationsOnDomain::from_vec_and_domain(expanded_domain.fft(&row_col), expanded_domain);
    end_timer!(interpolate_time);

    end_timer!(matrix_time);
    let evals_on_K = MatrixEvals {
        row: Cow::Owned(row_evals_on_K),
        col: Cow::Owned(col_evals_on_K),
        val: Cow::Owned(val_evals_on_K),
    };
    let evals_on_B = MatrixEvals {
        row: Cow::Owned(row_evals_on_B),
        col: Cow::Owned(col_evals_on_B),
        val: Cow::Owned(val_evals_on_B),
    };

    let m_name = matrix_name.to_string();
    MatrixArithmetization {
        row: LabeledPolynomial::new_owned(m_name.clone() + "_row", row, None, None),
        col: LabeledPolynomial::new_owned(m_name.clone() + "_col", col, None, None),
        val: LabeledPolynomial::new_owned(m_name.clone() + "_val", val, None, None),
        row_col: LabeledPolynomial::new_owned(m_name.clone() + "_row_col", row_col, None, None),
        evals_on_K,
        evals_on_B,
        row_col_evals_on_B: Cow::Owned(row_col_evals_on_B),
    }
}

fn is_in_ascending_order<T: Ord>(x_s: &[T], is_less_than: impl Fn(&T, &T) -> bool) -> bool {
    if x_s.is_empty() {
        true
    } else {
        let mut i = 0;
        let mut is_sorted = true;
        while i < (x_s.len() - 1) {
            is_sorted &= is_less_than(&x_s[i], &x_s[i + 1]);
            i += 1;
        }
        is_sorted
    }
}

/* ************************************************************************* */
/* ************************************************************************* */
/* ************************************************************************* */

pub(crate) struct ProverConstraintSystem<F: Field> {
    // Assignments of variables
    pub(crate) input_assignment: Vec<F>,
    pub(crate) witness_assignment: Vec<F>,
    pub(crate) num_input_variables: usize,
    pub(crate) num_witness_variables: usize,
    pub(crate) num_constraints: usize,
}

impl<F: Field> ProverConstraintSystem<F> {
    pub(crate) fn new() -> Self {
        Self {
            input_assignment: vec![F::one()],
            witness_assignment: Vec::new(),
            num_input_variables: 1usize,
            num_witness_variables: 0usize,
            num_constraints: 0usize,
        }
    }

    /// Formats the public input according to the requirements of the constraint
    /// system
    pub(crate) fn format_public_input(public_input: &[F]) -> Vec<F> {
        let mut input = vec![F::one()];
        input.extend_from_slice(public_input);
        input
    }

    /// Takes in a previously formatted public input and removes the formatting
    /// imposed by the constraint system.
    pub(crate) fn unformat_public_input(input: &[F]) -> Vec<F> {
        input[1..].to_vec()
    }

    pub(crate) fn make_matrices_square(&mut self) {
        let num_variables = self.num_input_variables + self.num_witness_variables;
        make_matrices_square(self, num_variables);
        assert_eq!(
            self.num_input_variables + self.num_witness_variables,
            self.num_constraints,
            "padding failed!"
        );
    }
}

impl<ConstraintF: Field> ConstraintSystem<ConstraintF> for ProverConstraintSystem<ConstraintF> {
    type Root = Self;

    #[inline]
    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<ConstraintF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_witness_variables;
        self.num_witness_variables += 1;

        self.witness_assignment.push(f()?);
        Ok(Variable::new_unchecked(VarIndex::Aux(index)))
    }

    #[inline]
    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<ConstraintF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_input_variables;
        self.num_input_variables += 1;

        self.input_assignment.push(f()?);
        Ok(Variable::new_unchecked(VarIndex::Input(index)))
    }

    #[inline]
    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _: LA, _: LB, _: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
        LB: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
        LC: FnOnce(LinearCombination<ConstraintF>) -> LinearCombination<ConstraintF>,
    {
        self.num_constraints += 1;
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn num_constraints(&self) -> usize {
        self.num_constraints
    }
}
