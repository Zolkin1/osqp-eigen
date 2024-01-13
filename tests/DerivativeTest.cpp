//
// Created by zolkin on 1/12/24.
//

#include <catch2/catch_all.hpp>

#include <OsqpEigen/OsqpEigen.h>

TEST_CASE("QPProblem - Unconstrained")
{
    constexpr double tolerance = 1e-4;

    constexpr int nv = 3;
    constexpr int nc = 1;

    double th1 = 1;
    double th2 = 2;
    double th3 = 3;

    Eigen::SparseMatrix<c_float> H_s(nv, nv);
    H_s.insert(0, 0) = th1;
    H_s.insert(1, 1) = th2;
    H_s.insert(2, 2) = th3;

    Eigen::Matrix<c_float, nv, 1> gradient;
    gradient << th1 + th2, 2, 4;

    Eigen::SparseMatrix<c_float> A(nc, nv);
    A.insert(0,0) = 1;

    Eigen::VectorXd lb(nc);
    lb << th3;

    Eigen::VectorXd ub(nc);
    ub << th3;

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(true);
    solver.settings()->setAlpha(1.0);

    solver.data()->setNumberOfVariables(nv);
    solver.data()->setNumberOfConstraints(nc);

    REQUIRE(solver.data()->setHessianMatrix(H_s));
    REQUIRE(solver.data()->setGradient(gradient));
    REQUIRE(solver.data()->setBounds(lb, ub));
    REQUIRE(solver.data()->setLinearConstraintsMatrix(A));

    REQUIRE(solver.initSolver());
    REQUIRE(solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError);

    // expected solution
    Eigen::Matrix<c_float, nv, 1> expectedSolution;
    expectedSolution << 3.000, -1.000, -1.3333;

    const Eigen::VectorXd xstar = solver.getSolution();

//    REQUIRE(solver.getSolution().isApprox(expectedSolution, tolerance));
    std::cout << "solution: " << solver.getSolution().transpose() << std::endl;


    /* Now take the derivative */
    Eigen::VectorXd dx(nv);
    dx = (H_s*xstar + gradient);
    std::cout << "dx: " << dx.transpose() << std::endl;

    Eigen::VectorXd dy_l(nc);
    dy_l << 0;
    Eigen::VectorXd dy_u(nc);
    dy_u << 0;
    solver.computeAdjointDerivative(dx, dy_l, dy_u);

    Eigen::VectorXd dq(nv);
    Eigen::VectorXd dl(nc);
    Eigen::VectorXd du(nc);
    solver.adjointDerivativeGetVec(dq, dl, du);

    dq = dq + xstar;

    std::cout << "dq: " << dq.transpose() << std::endl;
    std::cout << "dl: " << dl.transpose() << std::endl;
    std::cout << "du: " << du.transpose() << std::endl;

    Eigen::SparseMatrix<double> dA(nc, nv);
    Eigen::SparseMatrix<double> dP(nv, nv);
    solver.adjointDerivativeGetMat(dP, dA);

    dP = dP + 0.5*xstar*xstar.transpose();
    std::cout << "dP: \n" << dP.toDense() << std::endl;
    std::cout << "\ndA: \n" << dA.toDense() << std::endl;

    Eigen::SparseMatrix<double> dPdth2(nv, nv);
    dPdth2.insert(1,1) = 1;

    Eigen::MatrixXd dptemp = dP.cwiseProduct(dPdth2);

    double dhdth2 = 0;
    dhdth2 += dptemp.sum();

    Eigen::SparseMatrix<double> dAdth2(nc, nv);
    Eigen::MatrixXd datemp = dA.cwiseProduct(dAdth2);
    dhdth2 += datemp.sum();

    Eigen::VectorXd dgraddth2(nv);
    dgraddth2 << 1, 0, 0;
    dhdth2 += dq.dot(dgraddth2);

    Eigen::VectorXd dbdth2(nc);
    dhdth2 += dl.dot(dbdth2);

    std::cout << "dH/dth2: " << dhdth2 << std::endl;
}